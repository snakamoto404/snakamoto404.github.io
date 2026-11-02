export const BROWNIAN = Object.freeze({ initialVariance: 0.36, horizon: 3, steps: 240, particles: 22, xLimit: 4.5, floorY: 214 });
export const STACKED = Object.freeze({ particles: 300, steps: 20, radius: 1.9, pitch: 4.1, xLimit: 6.5, width: 280 });
const stackXScale = STACKED.width / (2 * STACKED.xLimit);
export const stackBinWidth = STACKED.pitch / stackXScale;

export const varianceAt = (time) => BROWNIAN.initialVariance + time;
export const gaussianPDF = (x, time) => Math.exp(-x * x / (2 * varianceAt(time))) / Math.sqrt(2 * Math.PI * varianceAt(time));
export const scorePosition = (initial, time) => initial * Math.sqrt(varianceAt(time) / BROWNIAN.initialVariance);

function normalCDF(x) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const tail = Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) * t *
    (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x >= 0 ? 1 - tail : tail;
}

export function initialParticles(count = BROWNIAN.particles) {
  return Float64Array.from({ length: count }, (_, index) => {
    const probability = (index + 0.5) / count;
    let low = -5, high = 5;
    for (let iteration = 0; iteration < 40; iteration++) {
      const middle = (low + high) / 2;
      if (normalCDF(middle) < probability) low = middle;
      else high = middle;
    }
    return (low + high) / 2 * Math.sqrt(BROWNIAN.initialVariance);
  });
}

export function brownianPaths(initial, seed = 0x2f61a97b) {
  let state = seed >>> 0;
  const random = () => {
    state += 0x6d2b79f5;
    let value = Math.imul(state ^ state >>> 15, state | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(Math.max(random(), 1e-12))) * Math.cos(2 * Math.PI * random());
  const stride = BROWNIAN.steps + 1;
  const paths = new Float64Array(initial.length * stride);
  const scale = Math.sqrt(BROWNIAN.horizon / BROWNIAN.steps);
  initial.forEach((start, particle) => {
    const offset = particle * stride;
    paths[offset] = start;
    for (let step = 1; step <= BROWNIAN.steps; step++) paths[offset + step] = paths[offset + step - 1] + scale * normal();
  });
  return paths;
}

export function brownianPosition(paths, particle, time) {
  const step = Math.max(0, Math.min(BROWNIAN.steps, time / BROWNIAN.horizon * BROWNIAN.steps));
  const lower = Math.floor(step), upper = Math.min(BROWNIAN.steps, lower + 1);
  const offset = particle * (BROWNIAN.steps + 1);
  return paths[offset + lower] + (paths[offset + upper] - paths[offset + lower]) * (step - lower);
}

export function project(x, time, height = 0) {
  const depth = time / BROWNIAN.horizon;
  return { x: 150 + x * (280 / (2 * BROWNIAN.xLimit)) * (1 - 0.18 * depth), y: BROWNIAN.floorY - 132 * depth - 52 * height };
}

export function marginalPoint(x, time) {
  const height = STACKED.particles * stackBinWidth * STACKED.pitch * gaussianPDF(x, time);
  return { x: 150 + x * stackXScale, y: BROWNIAN.floorY - height };
}

export function stackedMarginalFrames() {
  const initial = initialParticles(STACKED.particles);
  const paths = brownianPaths(initial, 0x75a21f49);
  const frames = [];
  let previous = null;
  for (let frame = 0; frame <= STACKED.steps; frame++) {
    const time = BROWNIAN.horizon * frame / STACKED.steps;
    const bins = new Map();
    initial.forEach((_, particle) => {
      const bin = Math.round(brownianPosition(paths, particle, time) / stackBinWidth);
      if (!bins.has(bin)) bins.set(bin, []);
      bins.get(bin).push(particle);
    });
    const positions = new Array(initial.length);
    for (const [bin, particles] of bins) {
      // Preserve vertical order when neighboring stacks exchange particles.
      particles.sort((a, b) => previous
        ? previous[a].row - previous[b].row || previous[a].bin - previous[b].bin || a - b
        : initial[a] - initial[b]);
      particles.forEach((particle, row) => {
        positions[particle] = { x: 150 + bin * STACKED.pitch, y: BROWNIAN.floorY - STACKED.radius - row * STACKED.pitch, bin, row };
      });
    }
    frames.push(positions);
    previous = positions;
  }
  return frames;
}

export function stackedMarginalState(frames, time) {
  const step = Math.max(0, Math.min(STACKED.steps, time / BROWNIAN.horizon * STACKED.steps));
  const lower = Math.floor(step), upper = Math.min(STACKED.steps, lower + 1);
  const phase = step - lower;
  const ease = (value) => { const u = Math.max(0, Math.min(1, value)); return u * u * (3 - 2 * u); };
  const mix = ease((phase - 0.45) / 0.45);
  const points = frames[lower].map((start, particle) => {
    const end = frames[upper][particle];
    const delay = ((particle * 0.61803398875) % 1) * 0.1;
    const u = lower === upper ? 0 : ease((phase - 0.45 - delay) / (0.45 - delay));
    const hop = Math.sin(Math.PI * u) * (2 + Math.min(7, Math.abs(end.x - start.x) * 0.16));
    return { x: start.x + (end.x - start.x) * u, y: start.y + (end.y - start.y) * u - hop };
  });
  return { points, time: BROWNIAN.horizon * (lower + mix) / STACKED.steps };
}

export function densityPath(time, perspective = false) {
  return Array.from({ length: 121 }, (_, index) => {
    const limit = perspective ? BROWNIAN.xLimit : STACKED.xLimit;
    const x = limit * (2 * index / 120 - 1);
    const point = perspective ? project(x, time, Math.exp(-x * x / (2 * varianceAt(time)))) : marginalPoint(x, time);
    return `${index ? 'L' : 'M'}${point.x.toFixed(2)} ${point.y.toFixed(2)}`;
  }).join(' ');
}

export function floorPaths(initial, paths, deterministic = false) {
  const steps = deterministic ? 48 : BROWNIAN.steps;
  return Array.from(initial, (start, particle) => Array.from({ length: steps + 1 }, (_, step) => {
    const time = BROWNIAN.horizon * step / steps;
    const x = deterministic ? scorePosition(start, time) : brownianPosition(paths, particle, time);
    const point = project(x, time);
    return `${step ? 'L' : 'M'}${point.x.toFixed(2)} ${point.y.toFixed(2)}`;
  }).join(' ')).join(' ');
}
