# Memory Management Simulator

A comprehensive **Memory Management Simulator** that models core operating system memory concepts including dynamic allocation, fragmentation, buddy allocation, caching, and virtual memory.

This project combines:
- An educational **Toy Chest analogy**
- A **Python CLI simulator**
- An **interactive HTML visualization**

---

## Table of Contents

- Project Overview
- Features
- Toy Chest Analogy
- System Architecture
- Memory Allocation Strategies
- Buddy System
- Cache Simulation
- Virtual Memory
- Implementation Details
- User Interfaces
- Testing and Validation
- Limitations
- Future Work
- Usage Examples
- References

---

## Project Overview

### Purpose
This project simulates how an operating system manages memory, including allocation, deallocation, fragmentation handling, caching, and virtual memory with paging.

### Components
- **Python CLI Simulator (`simulator.py`)** – Full-featured command-line simulator
- **HTML Visual UI (`index.html`)** – Interactive visualization of memory layout and fragmentation

---

## Features

- Dynamic memory allocation (First Fit, Best Fit, Worst Fit)
- Fragmentation visualization and metrics
- Buddy allocation system with coalescing
- Multilevel cache simulation (FIFO, LRU)
- Virtual memory with paging and page replacement
- CLI and web-based interactive interfaces

---

## Toy Chest Analogy

Memory is explained using a **toy chest** metaphor to make OS concepts intuitive.

| OS Concept | Toy Chest Analogy |
|-----------|------------------|
| Physical Memory | Toy chest slots |
| Allocation | Storing toys |
| Deallocation | Removing toys |
| Fragmentation | Holes in the chest |
| Cache | Pocket for favorite toys |
| Cache Hit | Toy in pocket (fast) |
| Cache Miss | Go to basement (slow) |
| Virtual Memory | Pretend larger chest |
| Page Fault | Fetch toy from closet |

---

## System Architecture

    ┌─────────────────────────────────────────────────┐
    │         Memory Management Simulator              │
    ├─────────────────────────────────────────────────┤
    │  Physical Memory     Buddy Allocator             │
    │                                                   │
    │  L1 Cache           L2 Cache                     │
    │                                                   │
    │  Virtual Memory (Paging & Translation)           │
    └─────────────────────────────────────────────────┘

### Execution Flow

    User Command → Allocator → Physical Memory
                         ↓
                      Cache
                         ↓
                   Virtual Memory
                         ↓
                  Statistics & UI

---

## Memory Allocation Strategies

### First Fit
- Allocates the first block large enough
- Fast and simple
- Can cause early fragmentation

### Best Fit
- Allocates the smallest possible block
- Reduces wasted space
- Slower, may create tiny fragments

### Worst Fit
- Allocates the largest available block
- Leaves large remainders
- Generally inefficient

---

## Buddy System

- Memory divided into power-of-two blocks
- Requests rounded up to nearest power of two
- Automatic merging with buddy on free

### Buddy Address Formula

    buddy_address = address XOR block_size

---

## Cache Simulation

### Cache Organization
- Set-associative cache
- Configurable size, block size, associativity
- FIFO and LRU replacement policies

### Address Format

    [ Tag ][ Set Index ][ Offset ]

### Replacement Policies

**FIFO**
- Evicts the oldest cache line

**LRU**
- Evicts the least recently used cache line

---

## Virtual Memory

- Paging-based virtual memory
- Demand paging
- FIFO or LRU page replacement

### Effective Access Time

    EAT = (1 − p) × memory_access_time + p × page_fault_time

---

## Implementation Details

### Core Modules

    MemorySimulator
    ├── PhysicalMemory
    ├── BuddyAllocator
    ├── Cache
    └── VirtualMemory

### Data Structures
- Memory blocks: Python list
- Cache: 2D list (sets × ways)
- Page table: Array/list

---

## User Interfaces

### Python CLI Commands

    init memory <size>
    set allocator <first_fit|best_fit|worst_fit>
    malloc <size>
    free <block_id>
    dump
    stats

    init_buddy <size>

    init_cache <size> <block_size> <associativity> <fifo|lru>
    cache_access <address>
    cache_stats

    init_vm <virtual> <physical> <page_size> [fifo|lru]
    translate <virtual_address>
    vm_stats

### HTML UI
- Visual memory blocks
- Terminal-style input
- Live statistics and fragmentation display
- Animations and demo mode

---

## Testing and Validation

### Sample Allocation Test

    init memory 1024
    malloc 100
    malloc 200
    malloc 300

Expected result:
- Three sequential allocated blocks

### Cache Test

    init_cache 512 64 4 lru
    cache_access 0x100
    cache_access 0x100
    cache_stats

Expected:
- One miss, one hit

### Virtual Memory Test

    init_vm 4096 1024 256 fifo
    translate 0x000
    translate 0x100
    translate 0x000

Expected:
- Two page faults, one page hit

---

## Limitations

- No real disk I/O simulation
- Single-process model
- Fixed page size
- No TLB simulation
- No NUMA support

---

## Future Work

- TLB simulation
- Clock and LFU replacement algorithms
- Multi-process memory isolation
- Memory-mapped files
- Cache and page-table visualization
- Performance benchmarking tools

---

## Usage Examples

### Memory Allocation

    init memory 1024
    set allocator first_fit
    malloc 200
    malloc 300
    free 1
    stats

### Cache Simulation

    init_cache 512 64 4 lru
    cache_access 0x100
    cache_stats

### Virtual Memory

    init_vm 4096 1024 256 lru
    translate 0x500
    vm_stats

---

