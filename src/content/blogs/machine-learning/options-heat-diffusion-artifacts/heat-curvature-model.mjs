export const CURVATURE = Object.freeze({ xMin: -1.4, xMax: 1.4, left: 26, right: 174, baseline: 92, yScale: 44, coefficient: 0.32 });

export function heatValue(kind, x, time) {
  const a = CURVATURE.coefficient;
  return kind === 'warming' ? 0.22 + a * x * x + a * time : 1.1 - a * x * x - a * time;
}

export function heatPoint(x, value) {
  return { x: CURVATURE.left + (x - CURVATURE.xMin) / (CURVATURE.xMax - CURVATURE.xMin) * (CURVATURE.right - CURVATURE.left), y: CURVATURE.baseline - CURVATURE.yScale * value };
}

export function heatCurve(kind, time) {
  return Array.from({ length: 61 }, (_, index) => {
    const x = CURVATURE.xMin + (CURVATURE.xMax - CURVATURE.xMin) * index / 60;
    const point = heatPoint(x, heatValue(kind, x, time));
    return `${index ? 'L' : 'M'}${point.x.toFixed(3)} ${point.y.toFixed(3)}`;
  }).join(' ');
}
