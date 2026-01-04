#!/usr/bin/env python3
"""
Memory Management Simulator
A comprehensive simulator for OS memory management including:
- Dynamic memory allocation (First Fit, Best Fit, Worst Fit)
- Buddy allocation system
- Multilevel cache simulation
- Virtual memory with paging
"""

import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math


class AllocationStrategy(Enum):
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"


class CachePolicy(Enum):
    FIFO = "fifo"
    LRU = "lru"


@dataclass
class MemoryBlock:
    """Represents a memory block"""
    start_addr: int
    size: int
    is_free: bool
    block_id: Optional[int] = None


@dataclass
class CacheLine:
    """Represents a cache line"""
    tag: int
    data: int
    valid: bool = False
    timestamp: int = 0
    frequency: int = 0


@dataclass
class PageTableEntry:
    """Page table entry for virtual memory"""
    valid: bool = False
    frame_number: int = -1
    referenced: bool = False
    modified: bool = False
    timestamp: int = 0


class PhysicalMemory:
    """Simulates physical memory with dynamic allocation"""
    
    def __init__(self, size: int):
        self.total_size = size
        self.blocks: List[MemoryBlock] = [MemoryBlock(0, size, True)]
        self.next_block_id = 1
        self.strategy = AllocationStrategy.FIRST_FIT
        self.allocations = 0
        self.deallocations = 0
        self.failed_allocations = 0
    
    def set_strategy(self, strategy: AllocationStrategy):
        self.strategy = strategy
    
    def malloc(self, size: int) -> Optional[Tuple[int, int]]:
        """Allocate memory block. Returns (block_id, address) or None"""
        if size <= 0:
            return None
        
        # Find suitable block based on strategy
        suitable_block = None
        block_index = -1
        
        if self.strategy == AllocationStrategy.FIRST_FIT:
            for i, block in enumerate(self.blocks):
                if block.is_free and block.size >= size:
                    suitable_block = block
                    block_index = i
                    break
        
        elif self.strategy == AllocationStrategy.BEST_FIT:
            best_size = float('inf')
            for i, block in enumerate(self.blocks):
                if block.is_free and block.size >= size and block.size < best_size:
                    suitable_block = block
                    block_index = i
                    best_size = block.size
        
        elif self.strategy == AllocationStrategy.WORST_FIT:
            worst_size = -1
            for i, block in enumerate(self.blocks):
                if block.is_free and block.size >= size and block.size > worst_size:
                    suitable_block = block
                    block_index = i
                    worst_size = block.size
        
        if suitable_block is None:
            self.failed_allocations += 1
            return None
        
        # Allocate block
        block_id = self.next_block_id
        self.next_block_id += 1
        
        if suitable_block.size == size:
            # Exact fit
            suitable_block.is_free = False
            suitable_block.block_id = block_id
        else:
            # Split block
            new_free_block = MemoryBlock(
                suitable_block.start_addr + size,
                suitable_block.size - size,
                True
            )
            suitable_block.size = size
            suitable_block.is_free = False
            suitable_block.block_id = block_id
            self.blocks.insert(block_index + 1, new_free_block)
        
        self.allocations += 1
        return (block_id, suitable_block.start_addr)
    
    def free(self, block_id: int) -> bool:
        """Free memory block and coalesce adjacent free blocks"""
        # Find block
        block_index = -1
        for i, block in enumerate(self.blocks):
            if not block.is_free and block.block_id == block_id:
                block_index = i
                break
        
        if block_index == -1:
            return False
        
        # Free the block
        self.blocks[block_index].is_free = True
        self.blocks[block_index].block_id = None
        self.deallocations += 1
        
        # Coalesce with adjacent free blocks
        self._coalesce(block_index)
        
        return True
    
    def _coalesce(self, index: int):
        """Coalesce adjacent free blocks"""
        # Merge with next block if free
        while index < len(self.blocks) - 1:
            if self.blocks[index].is_free and self.blocks[index + 1].is_free:
                self.blocks[index].size += self.blocks[index + 1].size
                self.blocks.pop(index + 1)
            else:
                break
        
        # Merge with previous block if free
        while index > 0:
            if self.blocks[index - 1].is_free and self.blocks[index].is_free:
                self.blocks[index - 1].size += self.blocks[index].size
                self.blocks.pop(index)
                index -= 1
            else:
                break
    
    def get_stats(self) -> Dict:
        """Calculate memory statistics"""
        used_memory = sum(b.size for b in self.blocks if not b.is_free)
        free_memory = self.total_size - used_memory
        
        # External fragmentation: free memory in non-contiguous blocks
        free_blocks = [b for b in self.blocks if b.is_free]
        if free_blocks:
            largest_free = max(b.size for b in free_blocks)
            external_frag = ((free_memory - largest_free) / self.total_size * 100) if free_memory > 0 else 0
        else:
            external_frag = 0
        
        return {
            'total_memory': self.total_size,
            'used_memory': used_memory,
            'free_memory': free_memory,
            'utilization': (used_memory / self.total_size * 100) if self.total_size > 0 else 0,
            'external_fragmentation': external_frag,
            'num_blocks': len(self.blocks),
            'allocations': self.allocations,
            'deallocations': self.deallocations,
            'failed_allocations': self.failed_allocations
        }
    
    def dump_memory(self):
        """Display memory layout"""
        for block in self.blocks:
            status = f"USED (id={block.block_id})" if not block.is_free else "FREE"
            end_addr = block.start_addr + block.size - 1
            print(f"[0x{block.start_addr:04X} - 0x{end_addr:04X}] {status} ({block.size} bytes)")


class BuddyAllocator:
    """Buddy memory allocation system"""
    
    def __init__(self, total_size: int):
        if not self._is_power_of_2(total_size):
            raise ValueError("Total size must be a power of 2")
        
        self.total_size = total_size
        self.min_block_size = 64  # Minimum allocation unit
        self.max_order = int(math.log2(total_size // self.min_block_size))
        
        # Free lists for each order
        self.free_lists: Dict[int, List[int]] = {i: [] for i in range(self.max_order + 1)}
        self.free_lists[self.max_order] = [0]  # Start with one large block
        
        # Track allocated blocks
        self.allocated_blocks: Dict[int, Tuple[int, int]] = {}  # block_id -> (address, order)
        self.next_block_id = 1
    
    def _is_power_of_2(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0
    
    def _get_order(self, size: int) -> int:
        """Get the order (power of 2) needed for size"""
        actual_size = max(size, self.min_block_size)
        order = math.ceil(math.log2(actual_size / self.min_block_size))
        return min(order, self.max_order)
    
    def _get_buddy_address(self, addr: int, order: int) -> int:
        """Calculate buddy address using XOR"""
        block_size = self.min_block_size * (2 ** order)
        return addr ^ block_size
    
    def malloc(self, size: int) -> Optional[Tuple[int, int]]:
        """Allocate memory using buddy system"""
        order = self._get_order(size)
        
        # Find free block of appropriate size
        actual_order = order
        while actual_order <= self.max_order:
            if self.free_lists[actual_order]:
                break
            actual_order += 1
        else:
            return None  # No block available
        
        # Get block from free list
        addr = self.free_lists[actual_order].pop(0)
        
        # Split blocks if necessary
        while actual_order > order:
            actual_order -= 1
            buddy_addr = self._get_buddy_address(addr, actual_order)
            self.free_lists[actual_order].append(buddy_addr)
        
        # Allocate block
        block_id = self.next_block_id
        self.next_block_id += 1
        self.allocated_blocks[block_id] = (addr, order)
        
        return (block_id, addr)
    
    def free(self, block_id: int) -> bool:
        """Free block and coalesce with buddy"""
        if block_id not in self.allocated_blocks:
            return False
        
        addr, order = self.allocated_blocks.pop(block_id)
        
        # Try to coalesce with buddy
        while order < self.max_order:
            buddy_addr = self._get_buddy_address(addr, order)
            
            if buddy_addr in self.free_lists[order]:
                # Buddy is free, coalesce
                self.free_lists[order].remove(buddy_addr)
                addr = min(addr, buddy_addr)
                order += 1
            else:
                break
        
        # Add to free list
        self.free_lists[order].append(addr)
        return True
    
    def dump_memory(self):
        """Display buddy allocator state"""
        print("\nBuddy Allocator State:")
        for order in range(self.max_order + 1):
            block_size = self.min_block_size * (2 ** order)
            free_count = len(self.free_lists[order])
            print(f"Order {order} (size {block_size}): {free_count} free blocks")


class Cache:
    """Multilevel cache simulation"""
    
    def __init__(self, size: int, block_size: int, associativity: int, policy: CachePolicy):
        self.size = size
        self.block_size = block_size
        self.associativity = associativity
        self.policy = policy
        
        self.num_sets = size // (block_size * associativity)
        self.cache: List[List[CacheLine]] = [
            [CacheLine(0, 0) for _ in range(associativity)]
            for _ in range(self.num_sets)
        ]
        
        self.hits = 0
        self.misses = 0
        self.accesses = 0
        self.current_time = 0
    
    def access(self, address: int) -> bool:
        """Access cache. Returns True on hit, False on miss"""
        self.accesses += 1
        self.current_time += 1
        
        # Calculate set and tag
        block_addr = address // self.block_size
        set_index = block_addr % self.num_sets
        tag = block_addr // self.num_sets
        
        cache_set = self.cache[set_index]
        
        # Check for hit
        for line in cache_set:
            if line.valid and line.tag == tag:
                self.hits += 1
                line.timestamp = self.current_time
                line.frequency += 1
                return True
        
        # Miss - need to insert
        self.misses += 1
        self._insert_line(set_index, tag)
        return False
    
    def _insert_line(self, set_index: int, tag: int):
        """Insert line into cache using replacement policy"""
        cache_set = self.cache[set_index]
        
        # Find invalid line first
        for line in cache_set:
            if not line.valid:
                line.valid = True
                line.tag = tag
                line.timestamp = self.current_time
                line.frequency = 1
                return
        
        # Need to evict
        victim_index = 0
        
        if self.policy == CachePolicy.FIFO:
            # Find oldest
            victim_index = min(range(len(cache_set)), key=lambda i: cache_set[i].timestamp)
        
        elif self.policy == CachePolicy.LRU:
            # Find least recently used
            victim_index = min(range(len(cache_set)), key=lambda i: cache_set[i].timestamp)
        
        # Replace victim
        cache_set[victim_index].tag = tag
        cache_set[victim_index].timestamp = self.current_time
        cache_set[victim_index].frequency = 1
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        hit_rate = (self.hits / self.accesses * 100) if self.accesses > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'accesses': self.accesses,
            'hit_rate': hit_rate
        }


class VirtualMemory:
    """Virtual memory system with paging"""
    
    def __init__(self, virtual_size: int, physical_size: int, page_size: int, policy: str = "fifo"):
        self.virtual_size = virtual_size
        self.physical_size = physical_size
        self.page_size = page_size
        self.policy = policy
        
        self.num_virtual_pages = virtual_size // page_size
        self.num_physical_frames = physical_size // page_size
        
        # Page table
        self.page_table: List[PageTableEntry] = [
            PageTableEntry() for _ in range(self.num_virtual_pages)
        ]
        
        # Frame table (which frames are free)
        self.free_frames: List[int] = list(range(self.num_physical_frames))
        self.used_frames: List[int] = []  # For FIFO
        
        self.page_faults = 0
        self.page_hits = 0
        self.current_time = 0
    
    def translate(self, virtual_addr: int) -> Optional[int]:
        """Translate virtual address to physical address"""
        if virtual_addr >= self.virtual_size:
            return None
        
        page_number = virtual_addr // self.page_size
        offset = virtual_addr % self.page_size
        
        pte = self.page_table[page_number]
        
        if pte.valid:
            # Page hit
            self.page_hits += 1
            pte.referenced = True
            pte.timestamp = self.current_time
            self.current_time += 1
            
            physical_addr = pte.frame_number * self.page_size + offset
            return physical_addr
        else:
            # Page fault
            self.page_faults += 1
            self._handle_page_fault(page_number)
            
            # Now translate
            pte = self.page_table[page_number]
            physical_addr = pte.frame_number * self.page_size + offset
            return physical_addr
    
    def _handle_page_fault(self, page_number: int):
        """Handle page fault with replacement policy"""
        if self.free_frames:
            # Allocate free frame
            frame = self.free_frames.pop(0)
        else:
            # Need to evict a page
            frame = self._select_victim()
        
        # Load page into frame
        self.page_table[page_number].valid = True
        self.page_table[page_number].frame_number = frame
        self.page_table[page_number].timestamp = self.current_time
        self.current_time += 1
        
        if frame not in self.used_frames:
            self.used_frames.append(frame)
    
    def _select_victim(self) -> int:
        """Select victim page for replacement"""
        if self.policy == "fifo":
            # Remove oldest frame
            frame = self.used_frames.pop(0)
            
            # Invalidate page table entry
            for pte in self.page_table:
                if pte.valid and pte.frame_number == frame:
                    pte.valid = False
                    break
            
            return frame
        
        elif self.policy == "lru":
            # Find least recently used
            oldest_time = float('inf')
            victim_frame = -1
            
            for pte in self.page_table:
                if pte.valid and pte.timestamp < oldest_time:
                    oldest_time = pte.timestamp
                    victim_frame = pte.frame_number
            
            # Invalidate
            for pte in self.page_table:
                if pte.valid and pte.frame_number == victim_frame:
                    pte.valid = False
                    break
            
            self.used_frames.remove(victim_frame)
            return victim_frame
        
        return 0
    
    def get_stats(self) -> Dict:
        """Get virtual memory statistics"""
        total = self.page_faults + self.page_hits
        hit_rate = (self.page_hits / total * 100) if total > 0 else 0
        
        return {
            'page_faults': self.page_faults,
            'page_hits': self.page_hits,
            'total_accesses': total,
            'hit_rate': hit_rate
        }


class MemorySimulator:
    """Main simulator orchestrating all components"""
    
    def __init__(self):
        self.memory: Optional[PhysicalMemory] = None
        self.buddy: Optional[BuddyAllocator] = None
        self.caches: List[Cache] = []
        self.virtual_memory: Optional[VirtualMemory] = None
        self.use_buddy = False
    
    def run_cli(self):
        """Run interactive command-line interface"""
        print("Memory Management Simulator")
        print("Type 'help' for available commands\n")
        
        while True:
            try:
                command = input("> ").strip()
                if not command:
                    continue
                
                self.process_command(command)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def process_command(self, command: str):
        """Process a single command"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "help":
            self.print_help()
        
        elif cmd == "init":
            if len(parts) < 3:
                print("Usage: init memory <size>")
                return
            size = int(parts[2])
            self.memory = PhysicalMemory(size)
            print(f"Initialized memory: {size} bytes")
        
        elif cmd == "init_buddy":
            if len(parts) < 2:
                print("Usage: init_buddy <size>")
                return
            size = int(parts[1])
            self.buddy = BuddyAllocator(size)
            self.use_buddy = True
            print(f"Initialized buddy allocator: {size} bytes")
        
        elif cmd == "set":
            if len(parts) < 3:
                print("Usage: set allocator <strategy>")
                return
            if parts[1] == "allocator" and self.memory:
                strategy = AllocationStrategy(parts[2])
                self.memory.set_strategy(strategy)
                print(f"Set allocation strategy: {parts[2]}")
        
        elif cmd == "malloc":
            if len(parts) < 2:
                print("Usage: malloc <size>")
                return
            size = int(parts[1])
            
            if self.use_buddy and self.buddy:
                result = self.buddy.malloc(size)
            elif self.memory:
                result = self.memory.malloc(size)
            else:
                print("Memory not initialized")
                return
            
            if result:
                block_id, addr = result
                print(f"Allocated block id={block_id} at address=0x{addr:04X}")
            else:
                print("Allocation failed: not enough memory")
        
        elif cmd == "free":
            if len(parts) < 2:
                print("Usage: free <block_id>")
                return
            block_id = int(parts[1])
            
            if self.use_buddy and self.buddy:
                success = self.buddy.free(block_id)
            elif self.memory:
                success = self.memory.free(block_id)
            else:
                print("Memory not initialized")
                return
            
            if success:
                print(f"Block {block_id} freed and merged")
            else:
                print(f"Block {block_id} not found")
        
        elif cmd == "dump":
            if self.use_buddy and self.buddy:
                self.buddy.dump_memory()
            elif self.memory:
                self.memory.dump_memory()
            else:
                print("Memory not initialized")
        
        elif cmd == "stats":
            if self.memory:
                stats = self.memory.get_stats()
                print(f"Total memory: {stats['total_memory']}")
                print(f"Used memory: {stats['used_memory']}")
                print(f"Free memory: {stats['free_memory']}")
                print(f"Utilization: {stats['utilization']:.2f}%")
                print(f"External fragmentation: {stats['external_fragmentation']:.2f}%")
                print(f"Allocations: {stats['allocations']}")
                print(f"Deallocations: {stats['deallocations']}")
                print(f"Failed allocations: {stats['failed_allocations']}")
        
        elif cmd == "init_cache":
            if len(parts) < 5:
                print("Usage: init_cache <size> <block_size> <associativity> <policy>")
                return
            size = int(parts[1])
            block_size = int(parts[2])
            assoc = int(parts[3])
            policy = CachePolicy(parts[4])
            cache = Cache(size, block_size, assoc, policy)
            self.caches.append(cache)
            print(f"Initialized L{len(self.caches)} cache")
        
        elif cmd == "cache_access":
            if len(parts) < 2:
                print("Usage: cache_access <address>")
                return
            addr = int(parts[1], 16) if parts[1].startswith("0x") else int(parts[1])
            
            for i, cache in enumerate(self.caches):
                hit = cache.access(addr)
                print(f"L{i+1} cache: {'HIT' if hit else 'MISS'}")
                if hit:
                    break
        
        elif cmd == "cache_stats":
            for i, cache in enumerate(self.caches):
                stats = cache.get_stats()
                print(f"\nL{i+1} Cache Statistics:")
                print(f"  Hits: {stats['hits']}")
                print(f"  Misses: {stats['misses']}")
                print(f"  Hit rate: {stats['hit_rate']:.2f}%")
        
        elif cmd == "init_vm":
            if len(parts) < 4:
                print("Usage: init_vm <virtual_size> <physical_size> <page_size> [policy]")
                return
            vsize = int(parts[1])
            psize = int(parts[2])
            page_size = int(parts[3])
            policy = parts[4] if len(parts) > 4 else "fifo"
            self.virtual_memory = VirtualMemory(vsize, psize, page_size, policy)
            print(f"Initialized virtual memory")
        
        elif cmd == "translate":
            if len(parts) < 2:
                print("Usage: translate <virtual_address>")
                return
            if not self.virtual_memory:
                print("Virtual memory not initialized")
                return
            
            vaddr = int(parts[1], 16) if parts[1].startswith("0x") else int(parts[1])
            paddr = self.virtual_memory.translate(vaddr)
            if paddr is not None:
                print(f"Virtual 0x{vaddr:04X} -> Physical 0x{paddr:04X}")
        
        elif cmd == "vm_stats":
            if self.virtual_memory:
                stats = self.virtual_memory.get_stats()
                print(f"Page faults: {stats['page_faults']}")
                print(f"Page hits: {stats['page_hits']}")
                print(f"Hit rate: {stats['hit_rate']:.2f}%")
        
        elif cmd == "exit" or cmd == "quit":
            sys.exit(0)
        
        else:
            print(f"Unknown command: {cmd}")
    
    def print_help(self):
        """Print available commands"""
        help_text = """
Available Commands:

Memory Allocation:
  init memory <size>              - Initialize physical memory
  init_buddy <size>               - Initialize buddy allocator (size must be power of 2)
  set allocator <strategy>        - Set strategy: first_fit, best_fit, worst_fit
  malloc <size>                   - Allocate memory block
  free <block_id>                 - Free memory block
  dump                            - Display memory layout
  stats                           - Show memory statistics

Cache Simulation:
  init_cache <size> <block> <assoc> <policy>  - Initialize cache level
  cache_access <address>          - Access cache with address
  cache_stats                     - Show cache statistics

Virtual Memory:
  init_vm <vsize> <psize> <page_size> [policy]  - Initialize virtual memory
  translate <virtual_address>     - Translate virtual to physical address
  vm_stats                        - Show virtual memory statistics

General:
  help                            - Show this help
  exit/quit                       - Exit simulator
"""
        print(help_text)


def main():
    simulator = MemorySimulator()
    simulator.run_cli()


if __name__ == "__main__":
    main()