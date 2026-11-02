export const CONVOLUTION = Object.freeze({ xMin: -6, xMax: 6, yMax: 2.6, pdfYMax: 0.65, left: 24, right: 696, top: 12, baseline: 252, initialVariance: 6 });
export const evaluationPoints = Object.freeze([-1, 2]);
export const normalPDF = (x) => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);

export function normalCDF(x) {
  const k = 1 / (1 + 0.2316419 * Math.abs(x));
  const tail = normalPDF(x) * k * (0.31938153 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))));
  return x >= 0 ? 1 - tail : tail;
}

// E[(x + sqrt(variance) Z)+], the Gaussian convolution of the terminal ReLU.
export function convolvedCall(x, variance) {
  if (variance <= 0) return Math.max(x, 0);
  const sigma = Math.sqrt(variance);
  return Math.max(x, 0, x * normalCDF(x / sigma) + sigma * normalPDF(x / sigma));
}

export const plotX = (x) => CONVOLUTION.left + (x - CONVOLUTION.xMin) / (CONVOLUTION.xMax - CONVOLUTION.xMin) * (CONVOLUTION.right - CONVOLUTION.left);
export const plotY = (y) => CONVOLUTION.baseline - y / CONVOLUTION.yMax * (CONVOLUTION.baseline - CONVOLUTION.top);
export const plotPDFY = (density) => CONVOLUTION.baseline - density / CONVOLUTION.pdfYMax * (CONVOLUTION.baseline - CONVOLUTION.top);

export function callCurve(variance) {
  return Array.from({ length: 241 }, (_, index) => {
    const x = CONVOLUTION.xMin + (CONVOLUTION.xMax - CONVOLUTION.xMin) * index / 240;
    return `${index ? 'L' : 'M'}${plotX(x).toFixed(3)} ${plotY(convolvedCall(x, variance)).toFixed(3)}`;
  }).join(' ');
}

export function gaussianCurve(mean, variance) {
  // A narrow finite kernel stands in for the limiting point mass at expiry.
  const sigma = Math.sqrt(Math.max(variance, 0.0009));
  const points = Array.from({ length: 161 }, (_, index) => {
    const z = -5 + 10 * index / 160;
    return { x: plotX(mean + sigma * z), y: plotPDFY(normalPDF(z) / sigma) };
  });
  const contour = points.map(({ x, y }, index) => `${index ? 'L' : 'M'}${x.toFixed(3)} ${y.toFixed(3)}`).join(' ');
  const fill = `M${points[0].x.toFixed(3)} ${CONVOLUTION.baseline} ${contour.replace(/^M/, 'L')} L${points.at(-1).x.toFixed(3)} ${CONVOLUTION.baseline}Z`;
  return { contour, fill };
}
