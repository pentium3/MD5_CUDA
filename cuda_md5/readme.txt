https://web.archive.org/web/20080421172400/http://majuric.org/software/cudamd5/

Notes: CUDA MD5 Hashing Experiments

Spurred by an interesting discussion at a New York City Area HPC Users Group meeting, I wrote a sample CUDA MD5 hash computation code (for words of 56 characters or shorter). It is a very straight forward GPU reimplementation of RSA's Md5.c code, with the goal of seeing how well would GPUs fare in cryptography/cryptanalysis applications.

The code in question can be found at http://majuric.org/software/cudamd5/source.

Benchmarks

I implemented two modes of operation. The first, MD5 calc, uses the GPU to quickly compute MD5 hashes given a (potentially large) dictionary of words. The computed hashes are stored to global device memory, and later transferred to the host. In my tests, I used a ~2M word "dictionary" (made by running the LaTeX source of one of my astrophysics papers through sed; see wordlist.txt in the source tarball).

The second, MD5 search, uses the GPU to search for a given MD5 hash in the dictionary. Similarly to MD5 calc, it computes the MD5 hash of each word in the dictionary, but then compares it to the target MD5 and sets a found flag in case of a match. In particular, note that the computed hashes are not transferred back to the host, nor stored in global device memory. As you will see below, this speeds things up quite a bit.

Here are a few results of running the code in both modes:

Device	MD5 search	MD5 calc
GeForce 8800 GTS	97.4	71.4
GeForce 8800 Ultra	155.9	108.6
GeForce 8400 GS	6.6	5.8
All numbers are in millions of MD5 hashes computed/searched per second (Mhash/sec). The timings are for the MD5 calculation step only, and do not include the time it takes to upload the dictionary to the device, or download the hashes back to host memory. The timings are averages of 10 runs, using the optimal number of threads per block (see the plots and discussion below).

For comparison, the same algorithm run on a single core of a CPU (g++ 4.1.2, with -O3 optimization flag) gives the following:

CPU	MD5 hash
Intel? Xeon? CPU 3.00GHz (single core)	3.9
Intel? Core?2 Quad CPU Q6700 @ 2.66GHz (single core)	4.3
Intel? Core?2 Quad CPU @ 2.40GHz (single code)	4.1
2x XEON 3.2GHz (DC + HT), MDcrack	42.3
The CPU results for hash and search modes should be very similar (although I haven't checked), so there's only a single column in this table. The last row is the score of a dual Xeon system running the highly optimized MDcrack code, taken from their website. This code is as fast as it gets (or at minimum a good approximation of it).

Performance Comparisons

What do these results show? The code on 8800 Ultra runs 36x faster than the same algorithm running on a single core of Q6700 @ 2.66GHz, or 9x faster compared to all four cores running simultaneously (and assuming the speedup scales linearly). The speedup over a single dual-core Xeon CPU running MD5crack is about 7x.

One way of visualizing this is noting that a single 8800 Ultra could brute-force break an MD5 hashed password of eight or less characters+numbers (A-Z, a-z, 0-9) in about ~16 days. You can imagine what a smarter (dictionary based) algorithm combined with a GPU cluster could do.

The numbers above could probably be improved on if one was so inclined (I personally am not). The MD5 code used here was written in less than 2 days, as a proof-of-concept, and with only a single one-liner GPU-specific optimization. The source is available so feel free to give it a shot.

Speed vs. Thread Count

A few interesting plots of speed vs. the number of threads per block.

Hash computation
 GeForce 8400 GS GeForce 8800 GTS GeForce 8800 Ultra GeForce 8800 GTS 512

Hash search
 GeForce 8400 GS GeForce 8800 GTS GeForce 8800 Ultra GeForce 8800 GTS 512

Notes:

Data for 8800 GTS 512 provided by Ale? Koval.

Observations:

Optimal numbers of threads per block are clearly identifiable.

This number is best determined experimentally (e.g., from plots such as the ones above), as it depends on the details of the algorithm, the actual device on which it's running, its memory access patterns, etc. It's difficult to get it right from theory only ¡ª e.g., note the significant difference between the best choices for calc and search modes, even though the algorithms are the same up until the very last line of their respective kernels.

The optimal number of threads per block turns out to be 63 for the calc algorithm. This is suspiciously close to 64, but is not 64 (in fact, with 64 threads per block the performance drops dramatically). Here's why: the multiprocessors are maximally utilized when a) the number of threads in the block is a multiple of 32 (the warp size), b) when the number of threads in a block is equal to at least a few warp sizes, and c) when there are multiple blocks sharing the same multiprocessor. However, as each thread requires 64 bytes of (dynamically allocated) shared memory allocated from a pool common to all threads running on the same MP, their number is limited by the total shared memory present (16384 bytes). Naively, given 16384 / (64*64) = 4, 64 threads per block should be perfect, fitting two blocks each having two warps, on a single MP. However, the devil is in the details: this calculation does not take into account a small per-block shared memory overhead (~32 bytes) reserved to pass the function parameters to the kernel. With that, the above case is actually becomes the worst case scenario, as only one block can be fitted to an MP, thus requiring serial execution of the second block while nearly half of the shared memory remains empty.

Author, Bugs, Notes, Etc.

Send any bug reports, questions, comments or notes to me, Mario Juric, <mjuric@ias.edu>

Last updated 2008-03-21 14:32:09 EDT