/* ============================================================
   NEUROFLAP — Simulación de un algoritmo genético
   ------------------------------------------------------------
   Idea general:

   1. Se crea una POBLACIÓN de pájaros. Cada pájaro tiene un
      "genoma": un puñado de números aleatorios (pesos) que
      definen cómo decide aletear.

   2. Todos los pájaros juegan a la vez (evaluación).
      Mientras más tiempo sobreviven y más tuberías pasan,
      más alto es su FITNESS (su "puntaje de éxito").

   3. Cuando todos mueren, se arma la siguiente GENERACIÓN:
        - Se elige a los padres mediante SELECCIÓN (los que
          tuvieron mejor fitness tienen más chances).
        - Se combinan sus genomas (CRUCE).
        - Se cambian algunos genes al azar (MUTACIÓN), para
          explorar nuevas soluciones.
        - El mejor pájaro de la generación anterior se copia
          tal cual (ELITISMO), para no perder el progreso.

   4. Se repite el proceso muchas generaciones. Con el tiempo,
      los genomas que sobreviven más se vuelven dominantes y
      los pájaros "aprenden" a jugar solos.
   ============================================================ */


// ==========================================
// 1. CONFIGURACIÓN GENERAL
// ==========================================

const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

const chart = document.getElementById("chart");
const chartCtx = chart.getContext("2d");

const WIDTH = canvas.width;
const HEIGHT = canvas.height;

const GROUND_HEIGHT = 45;
const BIRD_X = 145;
const BIRD_RADIUS = 10;

const PIPE_WIDTH = 62;
const PIPE_GAP = 142;
const PIPE_SPACING = 220;

const GRAVITY = 0.34;
const FLAP_POWER = -6.3;
const PIPE_SPEED = 2.5;

let populationSize = 30;
let simulationSpeed = 1;
let mutationRate = 0.10;

let birds = [];
let pipes = [];

let generation = 1;
let frame = 0;
let passedPipes = 0;

let bestEver = 0;
let bestThisGeneration = 0;

let paused = false;
let lastTime = 0;
let accumulator = 0;

const STEP_TIME = 1000 / 60;

let bestHistory = [];
let averageHistory = [];


// ==========================================
// 2. UTILIDADES
// ==========================================

function random(min, max) {
    return Math.random() * (max - min) + min;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}


// ==========================================
// 3. CEREBRO DEL PÁJARO
// ==========================================

/*
  Cada pájaro tiene una "neurona" muy simple que decide si
  aletear o no. Esta es la parte que el algoritmo genético
  va a ir optimizando con el tiempo.

  ENTRADAS (lo que el pájaro "ve"):
    0: su propia altura
    1: su velocidad vertical actual
    2: distancia horizontal hasta la próxima tubería
    3: a qué altura está el centro del hueco de esa tubería

  GENOMA (lo que hace único a cada pájaro):
    4 pesos, uno por cada entrada, más 1 sesgo ("bias") = 5 genes

  SALIDA:
    Se suman las entradas multiplicadas por sus pesos, se les
    aplica una función sigmoide (para que el resultado quede
    entre 0 y 1), y si el resultado supera 0.5, el pájaro
    aletea. Esto es básicamente una red neuronal de una sola
    neurona, sin entrenamiento por gradiente: el aprendizaje
    ocurre solo mediante selección, cruce y mutación.
*/

const GENE_COUNT = 5;

function createGenome() {
    return Array.from({ length: GENE_COUNT }, () => random(-1, 1));
}

function think(bird, pipe) {
    const inputs = [
        bird.y / HEIGHT,
        bird.velocity / 8,
        (pipe.x - BIRD_X) / WIDTH,
        pipe.gapY / HEIGHT
    ];

    let sum = bird.genome[4]; // el sesgo (bias)

    for (let i = 0; i < inputs.length; i++) {
        sum += inputs[i] * bird.genome[i];
    }

    const output = sigmoid(sum);

    return output > 0.5;
}


// ==========================================
// 4. CLASE PÁJARO
// ==========================================

class Bird {
    constructor(genome = createGenome()) {
        this.genome = [...genome];

        this.y = HEIGHT * 0.48;
        this.velocity = 0;

        this.alive = true;
        this.fitness = 0;

        this.pipesPassed = 0;
        this.color = `hsl(${random(42, 55)}, 95%, 58%)`;
    }

    flap() {
        this.velocity = FLAP_POWER;
    }

    update(pipe) {
        if (!this.alive) return;

        // Sobrevivir un fotograma más ya suma puntos de fitness.
        this.fitness += 1;

        if (think(this, pipe)) {
            this.flap();
        }

        this.velocity += GRAVITY;
        this.y += this.velocity;

        // Colisión con techo y suelo
        if (
            this.y - BIRD_RADIUS < 0 ||
            this.y + BIRD_RADIUS > HEIGHT - GROUND_HEIGHT
        ) {
            this.alive = false;
        }

        // Colisión con tuberías
        for (const p of pipes) {
            const withinX =
                BIRD_X + BIRD_RADIUS > p.x &&
                BIRD_X - BIRD_RADIUS < p.x + PIPE_WIDTH;

            const outsideGap =
                this.y - BIRD_RADIUS < p.gapY - PIPE_GAP / 2 ||
                this.y + BIRD_RADIUS > p.gapY + PIPE_GAP / 2;

            if (withinX && outsideGap) {
                this.alive = false;
            }
        }

        // Registrar tuberías superadas
        for (const p of pipes) {
            if (!p.counted && p.x + PIPE_WIDTH < BIRD_X) {
                p.counted = true;
                passedPipes++;
            }
        }

        // Bonificación de fitness por cada tubería superada.
        this.fitness += this.pipesPassed * 100;
    }
}


// ==========================================
// 5. OBSTÁCULOS
// ==========================================

function createPipe(x) {
    return {
        x,
        gapY: random(100, HEIGHT - GROUND_HEIGHT - 100),
        counted: false
    };
}

function resetPipes() {
    pipes = [
        createPipe(WIDTH + 80),
        createPipe(WIDTH + 80 + PIPE_SPACING),
        createPipe(WIDTH + 80 + PIPE_SPACING * 2)
    ];
}

function updatePipes() {
    for (const pipe of pipes) {
        pipe.x -= PIPE_SPEED;
    }

    if (pipes[0].x + PIPE_WIDTH < 0) {
        pipes.shift();

        const lastPipe = pipes[pipes.length - 1];
        pipes.push(createPipe(lastPipe.x + PIPE_SPACING));
    }
}


// ==========================================
// 6. CREAR POBLACIÓN
// ==========================================

function createPopulation() {
    birds = [];

    for (let i = 0; i < populationSize; i++) {
        birds.push(new Bird());
    }
}


// ==========================================
// 7. EVALUACIÓN (calcular el fitness de todos)
// ==========================================

function evaluatePopulation() {
    for (const bird of birds) {
        bird.fitness += bird.pipesPassed * 100;
    }

    // Ordenamos de mayor a menor fitness: birds[0] es el mejor.
    birds.sort((a, b) => b.fitness - a.fitness);

    bestThisGeneration = birds[0]?.fitness || 0;
    bestEver = Math.max(bestEver, bestThisGeneration);

    const average =
        birds.reduce((sum, b) => sum + b.fitness, 0) / Math.max(1, birds.length);

    bestHistory.push(bestThisGeneration);
    averageHistory.push(average);

    if (bestHistory.length > 50) {
        bestHistory.shift();
        averageHistory.shift();
    }
}


// ==========================================
// 8. SELECCIÓN
// ==========================================

/*
  Selección por TORNEO: se eligen unos pocos pájaros al azar
  (aquí, 4) y de ese grupo gana el que tenga mejor fitness.
  Así, los pájaros buenos tienen más probabilidad de ser
  elegidos como padres, pero los débiles todavía tienen una
  pequeña oportunidad (esto ayuda a mantener diversidad
  genética en la población).
*/

function tournamentSelection() {
    const tournamentSize = 4;
    let winner = null;

    for (let i = 0; i < tournamentSize; i++) {
        const candidate = birds[Math.floor(Math.random() * birds.length)];

        if (!winner || candidate.fitness > winner.fitness) {
            winner = candidate;
        }
    }

    return winner;
}


// ==========================================
// 9. CRUCE Y MUTACIÓN
// ==========================================

/*
  CRUCE (crossover): se toman dos padres y se arma un genoma
  nuevo combinando sus genes. Aquí cada gen se hereda al azar
  de uno de los dos padres (crossover uniforme).
*/
function crossover(parentA, parentB) {
    const child = [];

    for (let i = 0; i < GENE_COUNT; i++) {
        child[i] = Math.random() < 0.5 ? parentA.genome[i] : parentB.genome[i];
    }

    return child;
}

/*
  MUTACIÓN: con una probabilidad de "mutationRate" por gen,
  se le suma un pequeño valor aleatorio. Esto introduce
  variación nueva que el cruce por sí solo no podría generar,
  y evita que la población se estanque siempre con las mismas
  soluciones.
*/
function mutate(genome) {
    return genome.map(gene => {
        if (Math.random() < mutationRate) {
            gene += random(-0.5, 0.5);
        }

        return clamp(gene, -3, 3);
    });
}


// ==========================================
// 10. ARMAR LA SIGUIENTE GENERACIÓN
// ==========================================

function nextGeneration() {
    evaluatePopulation();

    const oldGeneration = birds;
    const newBirds = [];

    // ELITISMO: el mejor pájaro pasa intacto a la siguiente
    // generación, para no perder el progreso ya logrado.
    const elite = oldGeneration[0];
    newBirds.push(new Bird(elite.genome));

    // El resto de la población se genera combinando padres.
    while (newBirds.length < populationSize) {
        const parentA = tournamentSelection();
        const parentB = tournamentSelection();

        let childGenome = crossover(parentA, parentB);
        childGenome = mutate(childGenome);

        newBirds.push(new Bird(childGenome));
    }

    birds = newBirds;

    generation++;
    frame = 0;
    passedPipes = 0;

    resetPipes();
    updateUI();
}


// ==========================================
// 11. ACTUALIZAR LA SIMULACIÓN (un paso)
// ==========================================

function update() {
    frame++;

    updatePipes();

    const nextPipe = pipes.find(p => p.x + PIPE_WIDTH >= BIRD_X) || pipes[0];

    for (const bird of birds) {
        bird.update(nextPipe);
    }

    const aliveCount = birds.filter(bird => bird.alive).length;

    if (aliveCount === 0) {
        nextGeneration();
    }

    updateUI();
}


// ==========================================
// 12. DIBUJAR ESCENARIO
// ==========================================

function drawBackground() {
    const sky = ctx.createLinearGradient(0, 0, 0, HEIGHT);
    sky.addColorStop(0, "#a9ddeb");
    sky.addColorStop(1, "#e3f5e9");

    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, WIDTH, HEIGHT);

    // Nubes
    ctx.fillStyle = "#ffffff90";

    for (let i = 0; i < 5; i++) {
        const x = (i * 170 - frame * 0.3) % (WIDTH + 100);

        ctx.beginPath();
        ctx.arc(x, 75 + (i % 2) * 40, 18, 0, Math.PI * 2);
        ctx.arc(x + 22, 70 + (i % 2) * 40, 24, 0, Math.PI * 2);
        ctx.arc(x + 48, 78 + (i % 2) * 40, 17, 0, Math.PI * 2);
        ctx.fill();
    }

    // Colinas
    ctx.fillStyle = "#a5d5a7";
    ctx.beginPath();
    ctx.moveTo(0, HEIGHT - GROUND_HEIGHT);
    ctx.quadraticCurveTo(140, 280, 300, HEIGHT - GROUND_HEIGHT);
    ctx.quadraticCurveTo(470, 300, WIDTH, HEIGHT - GROUND_HEIGHT);
    ctx.fill();

    // Suelo
    ctx.fillStyle = "#c9a86a";
    ctx.fillRect(0, HEIGHT - GROUND_HEIGHT, WIDTH, GROUND_HEIGHT);

    ctx.fillStyle = "#78c66a";
    ctx.fillRect(0, HEIGHT - GROUND_HEIGHT, WIDTH, 9);

    ctx.strokeStyle = "#b08c4f";
    ctx.lineWidth = 2;

    for (let x = 0; x < WIDTH; x += 30) {
        const offset = (frame * 2) % 30;

        ctx.beginPath();
        ctx.moveTo(x - offset, HEIGHT - 20);
        ctx.lineTo(x + 10 - offset, HEIGHT - 12);
        ctx.stroke();
    }
}


// ==========================================
// 13. DIBUJAR TUBERÍAS
// ==========================================

function drawPipes() {
    for (const pipe of pipes) {
        const topHeight = pipe.gapY - PIPE_GAP / 2;
        const bottomY = pipe.gapY + PIPE_GAP / 2;

        // Cuerpo superior
        ctx.fillStyle = "#279b56";
        ctx.fillRect(pipe.x, 0, PIPE_WIDTH, topHeight);

        ctx.fillStyle = "#187b43";
        ctx.fillRect(pipe.x, 0, 7, topHeight);

        ctx.fillStyle = "#49c775";
        ctx.fillRect(pipe.x + PIPE_WIDTH - 8, 0, 8, topHeight);

        // Boca superior
        ctx.fillStyle = "#218c4d";
        ctx.fillRect(pipe.x - 6, topHeight - 17, PIPE_WIDTH + 12, 17);

        // Cuerpo inferior
        ctx.fillStyle = "#279b56";
        ctx.fillRect(pipe.x, bottomY, PIPE_WIDTH, HEIGHT - GROUND_HEIGHT - bottomY);

        ctx.fillStyle = "#187b43";
        ctx.fillRect(pipe.x, bottomY, 7, HEIGHT - GROUND_HEIGHT - bottomY);

        ctx.fillStyle = "#49c775";
        ctx.fillRect(
            pipe.x + PIPE_WIDTH - 8,
            bottomY,
            8,
            HEIGHT - GROUND_HEIGHT - bottomY
        );

        // Boca inferior
        ctx.fillStyle = "#218c4d";
        ctx.fillRect(pipe.x - 6, bottomY, PIPE_WIDTH + 12, 17);
    }
}


// ==========================================
// 14. DIBUJAR PÁJAROS
// ==========================================

function drawBird(bird, isBest) {
    if (!bird.alive) return;

    ctx.save();

    ctx.globalAlpha = isBest ? 1 : 0.42;

    const angle = clamp(bird.velocity * 0.06, -0.45, 0.65);

    ctx.translate(BIRD_X, bird.y);
    ctx.rotate(angle);

    // Cuerpo
    ctx.fillStyle = isBest ? "#57e389" : "#facc15";
    ctx.strokeStyle = isBest ? "#218b52" : "#a87919";
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.ellipse(0, 0, 13, 10, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Ala
    ctx.fillStyle = isBest ? "#a7f3c1" : "#fff0a0";

    ctx.beginPath();
    ctx.ellipse(-5, 2, 7, 4, -0.4, 0, Math.PI * 2);
    ctx.fill();

    // Pico
    ctx.fillStyle = "#f97316";

    ctx.beginPath();
    ctx.moveTo(10, -1);
    ctx.lineTo(20, 3);
    ctx.lineTo(10, 6);
    ctx.closePath();
    ctx.fill();

    // Ojo
    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(5, -4, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#172033";
    ctx.beginPath();
    ctx.arc(6, -4, 2, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
}


// ==========================================
// 15. RENDERIZADO GENERAL
// ==========================================

function draw() {
    drawBackground();
    drawPipes();

    for (const bird of birds) {
        drawBird(bird, false);
    }

    // Destacar al mejor pájaro vivo de la población.
    const bestAlive = birds.find(b => b.alive);

    if (bestAlive) {
        drawBird(bestAlive, true);
    }
}


// ==========================================
// 16. ESTADÍSTICAS E INTERFAZ
// ==========================================

function updateUI() {
    const aliveCount = birds.filter(bird => bird.alive).length;

    document.getElementById("generation").textContent = generation;
    document.getElementById("generationLabel").textContent =
        `Generación ${generation}`;

    document.getElementById("alive").textContent = aliveCount;
    document.getElementById("total").textContent = populationSize;
    document.getElementById("best").textContent = Math.round(bestEver);
    document.getElementById("pipes").textContent = passedPipes;

    document.getElementById("status").textContent = paused
        ? "Ⅱ PAUSADO"
        : "● EVOLUCIONANDO";

    const populationEl = document.getElementById("population");
    populationEl.innerHTML = "";

    birds.forEach((bird, index) => {
        const dot = document.createElement("div");
        dot.className = "bird-dot";

        if (!bird.alive) {
            dot.classList.add("dead");
        }

        if (index === 0 && bird.alive) {
            dot.classList.add("best");
        }

        dot.title = `Pájaro ${index + 1} · Fitness: ${Math.round(bird.fitness)}`;

        populationEl.appendChild(dot);
    });

    const info = document.getElementById("aiInfo");

    info.innerHTML = `
    <strong>Generación ${generation}</strong><br>
    Población viva: ${aliveCount} de ${populationSize}.<br>
    Mejor fitness de la generación: ${Math.round(bestThisGeneration)}.<br>
    Tasa de mutación: ${Math.round(mutationRate * 100)}%.
  `;

    drawChart();
}


// ==========================================
// 17. GRÁFICA DE EVOLUCIÓN
// ==========================================

function drawChart() {
    const rect = chart.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    chart.width = Math.max(1, rect.width * dpr);
    chart.height = Math.max(1, rect.height * dpr);

    chartCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = rect.width;
    const h = rect.height;

    chartCtx.clearRect(0, 0, w, h);

    const pad = 22;
    const graphW = w - pad * 2;
    const graphH = h - pad * 2;

    // Líneas de cuadrícula
    chartCtx.strokeStyle = "#293650";
    chartCtx.lineWidth = 1;

    for (let i = 0; i < 4; i++) {
        const y = pad + (graphH / 3) * i;

        chartCtx.beginPath();
        chartCtx.moveTo(pad, y);
        chartCtx.lineTo(w - pad, y);
        chartCtx.stroke();
    }

    if (bestHistory.length === 0) return;

    const maxValue = Math.max(10, ...bestHistory, ...averageHistory);

    function drawLine(data, color) {
        if (data.length < 1) return;

        chartCtx.beginPath();
        chartCtx.strokeStyle = color;
        chartCtx.lineWidth = 2.5;
        chartCtx.lineJoin = "round";
        chartCtx.lineCap = "round";

        data.forEach((value, i) => {
            const x =
                pad + (data.length === 1 ? 0 : (i / (data.length - 1)) * graphW);

            const y = pad + graphH - (value / maxValue) * graphH;

            if (i === 0) {
                chartCtx.moveTo(x, y);
            } else {
                chartCtx.lineTo(x, y);
            }
        });

        chartCtx.stroke();
    }

    drawLine(averageHistory, "#6ca8ff");
    drawLine(bestHistory, "#57e389");
}


// ==========================================
// 18. CONTROLES DE LA INTERFAZ
// ==========================================

document.getElementById("pauseBtn").addEventListener("click", () => {
    paused = !paused;

    document.getElementById("pauseBtn").textContent = paused
        ? "▶ Continuar"
        : "Ⅱ Pausar";

    updateUI();
});

document.getElementById("restartBtn").addEventListener("click", () => {
    resetSimulation();
});

document.getElementById("populationRange").addEventListener("input", event => {
    populationSize = Number(event.target.value);

    document.getElementById("populationValue").textContent = populationSize;

    resetSimulation();
});

document.getElementById("speedRange").addEventListener("input", event => {
    simulationSpeed = Number(event.target.value);

    document.getElementById("speedValue").textContent = `${simulationSpeed}x`;
});

document.getElementById("mutationRange").addEventListener("input", event => {
    mutationRate = Number(event.target.value) / 100;

    document.getElementById("mutationValue").textContent =
        `${Math.round(mutationRate * 100)}%`;
});


// ==========================================
// 19. REINICIAR SIMULACIÓN
// ==========================================

function resetSimulation() {
    generation = 1;
    frame = 0;
    passedPipes = 0;

    bestEver = 0;
    bestThisGeneration = 0;

    bestHistory = [];
    averageHistory = [];

    paused = false;

    document.getElementById("pauseBtn").textContent = "Ⅱ Pausar";

    createPopulation();
    resetPipes();
    updateUI();
}


// ==========================================
// 20. BUCLE PRINCIPAL DEL JUEGO
// ==========================================

function gameLoop(timestamp) {
    if (!lastTime) lastTime = timestamp;

    const delta = Math.min(timestamp - lastTime, 100);
    lastTime = timestamp;

    if (!paused) {
        accumulator += delta;

        let steps = 0;

        while (accumulator >= STEP_TIME && steps < 10) {
            for (let i = 0; i < simulationSpeed; i++) {
                update();
            }

            accumulator -= STEP_TIME;
            steps++;
        }
    }

    draw();

    requestAnimationFrame(gameLoop);
}


// ==========================================
// 21. INICIAR TODO
// ==========================================

createPopulation();
resetPipes();
updateUI();

requestAnimationFrame(gameLoop);