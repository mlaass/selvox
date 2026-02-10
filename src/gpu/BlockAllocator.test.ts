import { describe, it, expect } from 'vitest';
import { BlockAllocator } from './BlockAllocator.js';

describe('BlockAllocator', () => {
  it('first alloc returns offset 0', () => {
    const a = new BlockAllocator(100);
    expect(a.alloc(10)).toBe(0);
  });

  it('sequential allocs are contiguous', () => {
    const a = new BlockAllocator(100);
    expect(a.alloc(10)).toBe(0);
    expect(a.alloc(20)).toBe(10);
    expect(a.alloc(5)).toBe(30);
  });

  it('tracks used and available', () => {
    const a = new BlockAllocator(100);
    a.alloc(30);
    expect(a.used).toBe(30);
    expect(a.available).toBe(70);
    expect(a.capacity).toBe(100);
  });

  it('returns null when full', () => {
    const a = new BlockAllocator(10);
    expect(a.alloc(10)).toBe(0);
    expect(a.alloc(1)).toBeNull();
  });

  it('returns null when request exceeds available', () => {
    const a = new BlockAllocator(10);
    a.alloc(6);
    expect(a.alloc(5)).toBeNull();
  });

  it('free + re-alloc reuses space', () => {
    const a = new BlockAllocator(100);
    const off = a.alloc(10)!;
    a.free(off, 10);
    expect(a.available).toBe(100);
    expect(a.alloc(10)).toBe(0);
  });

  it('merges adjacent freed blocks', () => {
    const a = new BlockAllocator(100);
    const offA = a.alloc(10)!; // 0..9
    const offB = a.alloc(10)!; // 10..19
    a.alloc(10);               // 20..29

    a.free(offA, 10);
    a.free(offB, 10);
    // 0..19 should be merged into one free block (C still uses 10)
    expect(a.available).toBe(90);
    // Should be able to alloc 20 contiguous slots starting at 0
    expect(a.alloc(20)).toBe(0);
  });

  it('merges in reverse free order', () => {
    const a = new BlockAllocator(100);
    const offA = a.alloc(10)!;
    const offB = a.alloc(10)!;
    a.alloc(10);

    // Free B first, then A
    a.free(offB, 10);
    a.free(offA, 10);
    expect(a.alloc(20)).toBe(0);
  });

  it('handles fragmentation: reuses interior gap', () => {
    const a = new BlockAllocator(100);
    a.alloc(10);          // A: 0..9
    const offB = a.alloc(10)!; // B: 10..19
    a.alloc(10);          // C: 20..29

    a.free(offB, 10);     // gap at 10..19
    // Smaller alloc should fit in B's old slot
    expect(a.alloc(5)).toBe(10);
    expect(a.alloc(5)).toBe(15);
    // Gap fully consumed
    expect(a.alloc(1)).toBe(30);
  });

  it('rejects zero-size alloc', () => {
    const a = new BlockAllocator(100);
    expect(a.alloc(0)).toBeNull();
  });

  it('full cycle: alloc all, free all, alloc again', () => {
    const a = new BlockAllocator(50);
    a.alloc(50);
    expect(a.alloc(1)).toBeNull();
    a.free(0, 50);
    expect(a.available).toBe(50);
    expect(a.alloc(50)).toBe(0);
  });
});
