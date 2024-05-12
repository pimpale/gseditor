export function mod(n: number, m: number): number {
  return ((n % m) + m) % m;
}

export function clamp(v: number, min: number, max: number): number {
  return Math.max(Math.min(v, max), min)
}

export function logistic(x: number, x0: number, L: number, k: number): number {
  return L / (1 + Math.exp(-k * (x - x0)));
}

export function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const scores = logits.map(l => Math.exp(l - maxLogit));
  const denom = scores.reduce((a, b) => a + b);
  return scores.map(s => s / denom);
}

export function deg2rad(deg: number): number {
  return deg * Math.PI / 180;
}

export function arrayMin(arr: number[]| Float32Array): number {
  let min = Infinity;
  for (let i = 0; i < arr.length; i++) {
    min = Math.min(min, arr[i]);
  }
  return min;
}

export function arrayMax(arr: number[]| Float32Array|Uint32Array): number {
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    max = Math.max(max, arr[i]);
  }
  return max;
}