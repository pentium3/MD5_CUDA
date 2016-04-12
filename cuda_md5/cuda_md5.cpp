#include <string.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <valarray>

#include <cuda_runtime_api.h>
#include "cutil.h"
#include "util.h"

#define MD5_CPU				md5_cpu_v2
int niters = 10;

// Some declarations that should wind up in their own .h file at some point
void print_md5(uint *hash, bool crlf = true);
void md5_prep(char *c0);
double execute_kernel(int blocks_x, int blocks_y, int threads_per_block, int shared_mem_required, int realthreads, uint *gpuWords, uint *gpuHashes, bool search = false);
void init_constants(uint *target = NULL);
void md5_cpu(uint w[16], uint &a, uint &b, uint &c, uint &d);
void md5_cpu_v2(const uint *in, uint &a, uint &b, uint &c, uint &d);
int deviceQuery();

///////////////////////////////////////////////////////////
// CUDA helpers

//
// Find the dimensions (bx,by) of a 2D grid of blocks that 
// has as close to nblocks blocks as possible
//
void find_best_factorization(int &bx, int &by, int nblocks)
{
	bx = -1;
	int best_r = 100000;
	for(int bytmp = 1; bytmp != 65536; bytmp++)
	{
		int r  = nblocks % bytmp;
		if(r < best_r && nblocks / bytmp < 65535)
		{
			by = bytmp;
			bx = nblocks / bytmp;
			best_r = r;
			
			if(r == 0) { break; }
			bx++;
		}
	}
	if(bx == -1) { std::cerr << "Unfactorizable?!\n"; exit(-1); }
}

//
// Given a total number of threads, their memory requirements, and the
// number of threadsPerBlock, compute the optimal allowable grid dimensions.
// Returns false if the requested number of threads are impossible to fit to
// shared memory.
//
bool calculate_grid_parameters(int gridDim[3], int threadsPerBlock, int neededthreads, int dynShmemPerThread, int staticShmemPerBlock)
{
	const int shmemPerMP =  16384;

	int dyn_shared_mem_required = dynShmemPerThread*threadsPerBlock;
	int shared_mem_required = staticShmemPerBlock + dyn_shared_mem_required;
	if(shared_mem_required > shmemPerMP) { return false; }

	// calculate the total number of threads
	int nthreads = neededthreads;
	int over = neededthreads % threadsPerBlock;
	if(over) { nthreads += threadsPerBlock - over; } // round up to multiple of threadsPerBlock

	// calculate the number of blocks
	int nblocks = nthreads / threadsPerBlock;
	if(nthreads % threadsPerBlock) { nblocks++; }

	// calculate block dimensions so that there are as close to nblocks blocks as possible
	find_best_factorization(gridDim[0], gridDim[1], nblocks);
	gridDim[2] = 1;

	return true;
}

//
///////////////////////////////////////////////////////////


//
// Shared aux. functions (used both by GPU and CPU setup code)
//
union md5hash
{
	uint ui[4];
	char ch[16];
};

//
// Convert an array of null-terminated strings to an array of 64-byte words
// with proper MD5 padding
//
void md5_prep_array(std::valarray<char> &paddedWords, const std::vector<std::string> &words)
{
	paddedWords.resize(64*words.size());
	paddedWords = 0;

	for(int i=0; i != words.size(); i++)
	{
		char *w = &paddedWords[i*64];
		strncpy(w, words[i].c_str(), 56);
		md5_prep(w);
	}
}


//
// GPU calculation: given a vector ptext of plain text words, compute and
// return their MD5 hashes
//
int cuda_compute_md5s(std::vector<md5hash> &hashes, const std::vector<std::string> &ptext, uint *target = NULL, bool benchmark = false)
{
	CUT_DEVICE_INIT();

	// load the MD5 constant arrays into GPU constant memory
	init_constants(target);

	// pad dictionary words to 64 bytes (MD5 block size)
	std::valarray<char> paddedWords;
	md5_prep_array(paddedWords, ptext);

	// Upload the dictionary onto the GPU device
	uint *gpuWords, *gpuHashes = NULL;
	CUDA_SAFE_CALL( cudaMalloc((void **)&gpuWords, paddedWords.size()) );
	CUDA_SAFE_CALL( cudaMemcpy(gpuWords, &paddedWords[0], paddedWords.size(), cudaMemcpyHostToDevice) );

	if(target != NULL)
	{
		// allocate GPU memory for match signal, instead of the actual hashes
		CUDA_SAFE_CALL( cudaMalloc((void **)&gpuHashes, 4*sizeof(uint)) );
		uint tmp[4] = {0}; // initialize to zero
		CUDA_SAFE_CALL( cudaMemcpy(gpuHashes, tmp, 4*sizeof(uint), cudaMemcpyHostToDevice) );
	}
	else
	{
		// allocate GPU memory for computed hashes
		CUDA_SAFE_CALL( cudaMalloc((void **)&gpuHashes, 4*sizeof(uint)*ptext.size()) );
	}

	//
	// The icky part: compute the optimal number of threads per block,
	// and the number of blocks
	//
	int dynShmemPerThread = 64;	// built in the algorithm
	int staticShmemPerBlock = 32;	// read from .cubin file

	double bestTime = 1e10, bestRate = 0.;
	int bestThreadsPerBlock;
	int nthreads = ptext.size();
	int tpb = benchmark ? 10 : 63;	// tpb is number of threads per block
					// 63 is the experimentally determined best case scenario on my 8800 GTX Ultra
	do
	{
		int gridDim[3];
		if(!calculate_grid_parameters(gridDim, tpb, nthreads, dynShmemPerThread, staticShmemPerBlock)) { continue; }

		// Call the kernel 10 times and calculate the average running time
		double gpuTime = 0.; int k;
		for(k=0; k != niters; k++)
		{
			gpuTime += execute_kernel(gridDim[0], gridDim[1], tpb, tpb*dynShmemPerThread, ptext.size(), gpuWords, gpuHashes, target != NULL);
		}
		gpuTime /= k;
		double rate = 1000 * ptext.size() / gpuTime;

		if(bestRate < rate)
		{
			bestTime = gpuTime;
			bestRate = rate;
			bestThreadsPerBlock = tpb;
		}

		if(benchmark)
		{
			std::cout << "words=" << ptext.size()
				<< " tpb=" << tpb
				<< " nthreads=" << gridDim[0]*gridDim[1]*tpb << " nblocks=" << gridDim[0]*gridDim[1]
				<< " gridDim[0]=" << gridDim[0] << " gridDim[1]=" << gridDim[1]
				<< " padding=" << gridDim[0]*gridDim[1]*tpb - ptext.size()
				<< " dynshmem=" << dynShmemPerThread*tpb
				<< " shmem=" << staticShmemPerBlock + dynShmemPerThread*tpb
				<< " gpuTime=" << gpuTime
				<< " rate=" << (int)rint(rate)
				<< std::endl;
		}

	} while(benchmark && tpb++ <= 512);

	if(benchmark)
	{
		std::cerr << "\nBest case: threadsPerBlock=" << bestThreadsPerBlock << "\n";
	}
	std::cerr << "GPU MD5 time : " <<  bestTime << " ms (" << std::fixed << 1000. * ptext.size() / bestTime << " hash/second)\n";

	// Download the results
	if(target != NULL)
	{
		uint ret[4];
		CUDA_SAFE_CALL( cudaMemcpy(ret, gpuHashes, sizeof(uint)*4, cudaMemcpyDeviceToHost) );
		return ret[3] ? ret[0] : -1;
	}
	else
	{
		// Download the computed hashes
		hashes.resize(ptext.size());
		CUDA_SAFE_CALL( cudaMemcpy(&hashes.front(), gpuHashes, sizeof(uint)*4*ptext.size(), cudaMemcpyDeviceToHost) );
	}

	// Shutdown
	CUDA_SAFE_CALL( cudaFree(gpuWords) );
	CUDA_SAFE_CALL( cudaFree(gpuHashes) );

	return 0;
}

//
// CPU calculation: given a vector ptext of plain text words, compute and
// return their MD5 hashes
//
void cpu_compute_md5s(std::vector<md5hash> &hashes, const std::vector<std::string> &ptext)
{
	std::valarray<char> paddedWords;
	md5_prep_array(paddedWords, ptext);

	hashes.resize(ptext.size());

	unsigned int hTimer;
	CUT_SAFE_CALL( cutCreateTimer(&hTimer) );
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
	CUT_SAFE_CALL( cutStartTimer(hTimer) );

	for(int i=0; i != hashes.size(); i++)
	{
		MD5_CPU(((uint*)&paddedWords[0])+16*i, hashes[i].ui[0], hashes[i].ui[1], hashes[i].ui[2], hashes[i].ui[3]);
	}

	CUT_SAFE_CALL( cutStopTimer(hTimer) );
	double cpuTime = cutGetTimerValue(hTimer);
	CUT_SAFE_CALL( cutDeleteTimer( hTimer) );
	std::cerr << "CPU MD5 time : " <<  cpuTime << " ms (" << std::fixed << 1000. * ptext.size() / cpuTime << " hash/second)\n";

}

md5hash single_md5(std::string &ptext)
{
	md5hash h;

	char w[64] = {0};
	strncpy(w, ptext.c_str(), 56);
	md5_prep(w);
	MD5_CPU((uint*)&w[0], h.ui[0], h.ui[1], h.ui[2], h.ui[3]);

	return h;
}

//
// Compare and print the MD5 hashes hashes1 and hashes2 of plaintext vector
// ptext. Complain if they don't match.
//
void compare_hashes(std::vector<md5hash> &hashes1, std::vector<md5hash> &hashes2, const std::vector<std::string> &ptext)
{
	// Compare & print
	for(int i=0; i != hashes1.size(); i++)
	{
		if(memcmp(hashes1[i].ui, hashes2[i].ui, 16) == 0)
		{
			//printf("OK   ");
			//print_md5(hashes1[i].ui);
		}
		else
		{
			printf("%-56s ", ptext[i].c_str());
			printf("ERROR   ");
			print_md5(hashes1[i].ui, false);
			printf(" != ");
			print_md5(hashes2[i].ui);
			std::cerr << "Hash " << i << " didn't match. Test failed. Aborting.\n";
			return;
		}
	}
	std::cerr << "All hashes match.\n";
}

int main(int argc, char **argv)
{
	option_reader o;

	bool devQuery = false, benchmark = false;
	std::string target_word;

	o.add("deviceQuery", devQuery, option_reader::flag);
	o.add("benchmark", benchmark, option_reader::flag);
	o.add("search", target_word, option_reader::optparam);
	o.add("benchmark-iters", niters, option_reader::optparam);

	if(!o.process(argc, argv))
	{
		std::cerr << "Usage: " << o.cmdline_usage(argv[0]) << "\n";
		return -1;
	}

	if(devQuery) { return deviceQuery(); }


	// Load plaintext dictionary
	std::vector<std::string> ptext;
	std::cerr << "Loading words from stdin ...\n";
	std::string word;
	while(std::cin >> word)
	{
		ptext.push_back(word);
	}
	std::cerr << "Loaded " << ptext.size() << " words.\n\n";

	// Do search/calculation
	std::vector<md5hash> hashes_cpu, hashes_gpu;
	if(!target_word.empty())
	{
		// search for a given target word
		md5hash target = single_md5(target_word);

		int match = cuda_compute_md5s(hashes_gpu, ptext, target.ui, benchmark);
		if(match >= 0)
		{
			std::cerr << "Found match: index=" << match << " " << " word='" << ptext[match] << "'\n";
		}
		else
		{
			std::cerr << "Match not found.\n";
		}
	}
	else
	{
		// Compute hashes
		cuda_compute_md5s(hashes_gpu, ptext, NULL, benchmark);
		cpu_compute_md5s(hashes_cpu, ptext);

		// Verify the answers
		compare_hashes(hashes_gpu, hashes_cpu, ptext);
	}

	return 0;
}
