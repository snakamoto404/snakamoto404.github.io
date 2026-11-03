// Normalized free Gaussian packet, in units m = hbar = 1.
export const FREE_PARTICLE = Object.freeze({
  width: 720, height: 232, left: 24, right: 696, baseline: 142,
  xMin: -9, xMax: 9, initialCenter: -3, initialSigma: 0.9,
  momentum: 3, duration: 2, scale: 110, densityScale: 140, samples: 720,
});

export function wavefunction(x, time) {
  const { initialCenter, initialSigma, momentum } = FREE_PARTICLE;
  const variance = initialSigma ** 2;
  const spreading = time / (2 * variance);
  const widthFactor = 1 + spreading ** 2;
  const displacement = x - initialCenter - momentum * time;
  const amplitude = (2 * Math.PI * variance * widthFactor) ** -0.25
    * Math.exp(-(displacement ** 2) / (4 * variance * widthFactor));
  const phase = momentum * (x - initialCenter) - momentum ** 2 * time / 2
    + spreading * displacement ** 2 / (4 * variance * widthFactor)
    - Math.atan(spreading) / 2;
  return {
    real: amplitude * Math.cos(phase),
    imaginary: amplitude * Math.sin(phase),
    density: amplitude ** 2,
  };
}

export const plotX = (x) => FREE_PARTICLE.left
  + (x - FREE_PARTICLE.xMin) / (FREE_PARTICLE.xMax - FREE_PARTICLE.xMin)
  * (FREE_PARTICLE.right - FREE_PARTICLE.left);
const plotY = (value, scale) => FREE_PARTICLE.baseline - scale * value;

export function wavepacketFrame(time) {
  const paths = { real: '', imaginary: '', density: '' };
  const center = FREE_PARTICLE.initialCenter + FREE_PARTICLE.momentum * time;
  const sigma = Math.hypot(FREE_PARTICLE.initialSigma, time / (2 * FREE_PARTICLE.initialSigma));
  let densityStart = FREE_PARTICLE.left, densityEnd = FREE_PARTICLE.right;
  for (let index = 0; index <= FREE_PARTICLE.samples; index++) {
    const x = FREE_PARTICLE.xMin + index / FREE_PARTICLE.samples
      * (FREE_PARTICLE.xMax - FREE_PARTICLE.xMin);
    const sample = wavefunction(x, time);
    for (const key of ['real', 'imaginary', 'density']) {
      // Omit subpixel tails so the strokes do not repaint the whole axis.
      if (Math.abs(x - center) > (key === 'density' ? 3.5 : 4.5) * sigma) continue;
      const prefix = paths[key] ? 'L' : 'M';
      const scale = key === 'density' ? FREE_PARTICLE.densityScale : FREE_PARTICLE.scale;
      if (key === 'density') {
        if (!paths.density) densityStart = plotX(x);
        densityEnd = plotX(x);
      }
      paths[key] += `${prefix}${plotX(x).toFixed(2)} ${plotY(sample[key], scale).toFixed(2)}`;
    }
  }
  return {
    ...paths,
    fill: `${paths.density}L${densityEnd.toFixed(2)} ${FREE_PARTICLE.baseline}H${densityStart.toFixed(2)}Z`,
  };
}
