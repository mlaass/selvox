export class RingBuffer {
  private buf: Float64Array;
  private head = 0;
  private count = 0;

  constructor(public readonly capacity: number) {
    this.buf = new Float64Array(capacity);
  }

  push(value: number): void {
    this.buf[this.head] = value;
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) this.count++;
  }

  /** Read by logical index: 0 = oldest entry. */
  get(i: number): number {
    const start = this.count < this.capacity ? 0 : this.head;
    return this.buf[(start + i) % this.capacity];
  }

  get length(): number {
    return this.count;
  }

  minMax(): { min: number; max: number } {
    if (this.count === 0) return { min: 0, max: 0 };
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < this.count; i++) {
      const v = this.get(i);
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return { min, max };
  }
}
