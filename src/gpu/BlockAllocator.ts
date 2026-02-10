interface FreeBlock {
  offset: number;
  size: number;
}

export class BlockAllocator {
  private _capacity: number;
  private _used: number = 0;
  private freeList: FreeBlock[];

  constructor(capacity: number) {
    this._capacity = capacity;
    this.freeList = [{ offset: 0, size: capacity }];
  }

  /** Allocate `size` contiguous slots. Returns start offset, or null if OOM. */
  alloc(size: number): number | null {
    if (size <= 0) return null;

    // First-fit search
    for (let i = 0; i < this.freeList.length; i++) {
      const block = this.freeList[i];
      if (block.size >= size) {
        const offset = block.offset;
        if (block.size === size) {
          this.freeList.splice(i, 1);
        } else {
          block.offset += size;
          block.size -= size;
        }
        this._used += size;
        return offset;
      }
    }
    return null;
  }

  /** Free `size` slots starting at `offset`. Merges adjacent free blocks. */
  free(offset: number, size: number): void {
    if (size <= 0) return;
    this._used -= size;

    // Find insertion index (keep sorted by offset)
    let idx = 0;
    while (idx < this.freeList.length && this.freeList[idx].offset < offset) {
      idx++;
    }

    this.freeList.splice(idx, 0, { offset, size });

    // Merge with next neighbor
    if (idx + 1 < this.freeList.length) {
      const curr = this.freeList[idx];
      const next = this.freeList[idx + 1];
      if (curr.offset + curr.size === next.offset) {
        curr.size += next.size;
        this.freeList.splice(idx + 1, 1);
      }
    }

    // Merge with previous neighbor
    if (idx > 0) {
      const prev = this.freeList[idx - 1];
      const curr = this.freeList[idx];
      if (prev.offset + prev.size === curr.offset) {
        prev.size += curr.size;
        this.freeList.splice(idx, 1);
      }
    }
  }

  get used(): number {
    return this._used;
  }

  get available(): number {
    return this._capacity - this._used;
  }

  get capacity(): number {
    return this._capacity;
  }
}
