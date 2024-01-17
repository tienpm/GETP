/*
 * mm.c - The fastest, least memory-efficient malloc package.
 *
 * Description of my solutions:
 * In this naive approach, a block is allocated by simply incrementing
 * the brk pointer.  A block is pure payload. There are no headers or
 * footers.  Blocks are never coalesced or reused. Realloc is
 * implemented directly using mm_malloc and mm_free.
 *
 * Implemented with an explicit free list allocator to manage allocation of and
 * free of memory, next fit algorithm to
 *
 * Allocated Block          Free Block
 *  ---------               ---------
 * | HEADER  |             | HEADER  |
 *  ---------               ---------
 * |         |             |  NEXT   |
 * |         |              ---------
 * | PAYLOAD |             |  PREV   |
 * |         |              ---------
 * |         |             |         |
 *  ---------              |         |
 * | FOOTER  |              ---------
 *  ---------              | FOOTER  |
 *                          ---------
 *
 * Free list organization:
 * Free blocks on the heap are organized using an explicit free list with the
 * head of the list being pointed to by a pointer free_listp (see diagram below
 * in mm_init). Each free block contains two pointers, one pointing to the next
 * free block, and one pointing to the previous free block. The minimum payload
 * for a free block must be 8 bytes to support the two pointers. The overall
 * size of a free block is then 16 bytes, which includes the 4 byte header and 4
 * byte footer.
 *
 * Free list manipulation:
 * The free list is maintained as a doubly linked list. Free blocks are removed
 * using a doubly linked list removal strategy and then coalesced to merge any
 * adjacent free blocks. Free blocks are added to the list using a LIFO
 * insertion policy. Each free block is added to the front of the free list. For
 * more information on how the free list is modified, see the functions
 * 'remove_freeblock' and 'coalesce'.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memlib.h"
#include "mm.h"

/*********************************************************
 * NOTE TO STUDENTS: Before you do anything else, please
 * provide your team information in the following struct.
 ********************************************************/
team_t team = {
    /* Team name */
    "getp16",
    /* First member's full name */
    "Tien Pham",
    /* First member's email address */
    "tien.pham@moreh.com.vn",
    /* Second member's full name (leave blank if none) */
    "",
    /* Second member's email address (leave blank if none) */
    ""};

/* single word (4) or double word (8) alignment */
#define ALIGNMENT 8

/* rounds up to the nearest multiple of ALIGNMENT */
#define ALIGN(size) (((size) + (ALIGNMENT - 1)) & ~0x7)

#define SIZE_T_SIZE (ALIGN(sizeof(size_t)))

// My MACRO

#define MAX(x, y) ((x) > (y) ? (x) : (y))

/* Pack a size and allocated bit into a word */
#define PACK(size, alloc) ((size) | (alloc)) // line:vm:mm:pack

/* Read and write a word at address p */
#define GET(p) (*(unsigned int *)(p))              // line:vm:mm:get
#define PUT(p, val) (*(unsigned int *)(p) = (val)) // line:vm:mm:put

/* Read the size and allocated fields from address p */
#define GET_SIZE(p) (GET(p) & ~0x7) // line:vm:mm:getsize
#define GET_ALLOC(p) (GET(p) & 0x1) // line:vm:mm:getalloc

/* Given block ptr bp, compute address of its header and footer */
#define HDRP(bp) ((char *)(bp)-WSIZE)                        // line:vm:mm:hdrp
#define FTRP(bp) ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE) // line:vm:mm:ftrp

/* Given block ptr bp, compute address of next and previous blocks */
#define NEXT_BLKP(bp)                                                          \
  ((char *)(bp) + GET_SIZE(((char *)(bp)-WSIZE))) // line:vm:mm:nextblkp
#define PREV_BLKP(bp)                                                          \
  ((char *)(bp)-GET_SIZE(((char *)(bp)-DSIZE))) // line:vm:mm:prevblkp
/* $end mallocmacros */

/* Global variables */
static char *heap_listp = 0; /* Pointer to first block */
#ifdef NEXT_FIT
static char *rover;

#define WSIZE 4
#define WDSIZE 8
#define CHUNK_SIZE (1 << 12)

// Private variables represeneting the heap and free list within the heap
static char *heap_list_ptr = 0;
static char *free_lis_ptr = 0;

/* Function prototypes for internal helper routines */
static void *extend_heap(size_t words);
static void place(void *bp, size_t asize);
static void *find_fit(size_t asize);
static void *coalesce(void *bp);
static void printblock(void *bp);
static void checkheap(int verbose);
static void checkblock(void *bp);

/*
 * mm_init - initialize the malloc package.
 */
int mm_init(void) {
  if (heap_list_ptr == mem_sbrk(4 * WSIZE == (void *)-1))
    return -1;

  *(heap_list_ptr) = 0;                           // Alignment padding
  *(heap_list_ptr + (1 * WSIZE)) = ALIGNMENT | 1; // Prologue header
  *(heap_list_ptr + (2 * WSIZE)) = ;              // Prologue footer
  *(heap_list_ptr) + (3 * WSIZE) = ;              // Epilogue header

  // Extend the empty heap with free block of CHUNK_SIZE bytes

  return 0;
}

/*
 * mm_malloc - Allocate a block by incrementing the brk pointer.
 *     Always allocate a block whose size is a multiple of the alignment.
 */
void *mm_malloc(size_t size) {
  int newsize = ALIGN(size + SIZE_T_SIZE);
  void *p = mem_sbrk(newsize);
  if (p == (void *)-1)
    return NULL;
  else {
    *(size_t *)p = size;
    return (void *)((char *)p + SIZE_T_SIZE);
  }
}

/*
 * mm_free - Freeing a block does nothing.
 */
void mm_free(void *ptr) {}

/*
 * mm_realloc - Implemented simply in terms of mm_malloc and mm_free
 */
void *mm_realloc(void *ptr, size_t size) {
  void *oldptr = ptr;
  void *newptr;
  size_t copySize;

  newptr = mm_malloc(size);
  if (newptr == NULL)
    return NULL;
  copySize = *(size_t *)((char *)oldptr - SIZE_T_SIZE);
  if (size < copySize)
    copySize = size;
  memcpy(newptr, oldptr, copySize);
  mm_free(oldptr);
  return newptr;
}
