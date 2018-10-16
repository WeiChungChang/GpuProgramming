#include <stdio.h>
#include <cuda.h>

#define HANDLE_ERROR(apiFuncCall)                                              \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define HANDLE_NULL(x)
#define TOTAL 1024

float cuda_malloc_test( int size, bool up ) {
    cudaEvent_t start, stop;
    int *a, *dev_a;
    float elapsedTime = 0.0f;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    a = (int*)malloc( size * sizeof( *a ) );
    if (!a) {
        exit(-1);
    }
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a,
    size * sizeof( *dev_a ) ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    for (int i=0; i<TOTAL; i++) {
    if (up)
        HANDLE_ERROR( cudaMemcpy( dev_a, a, size * sizeof( *dev_a ),cudaMemcpyHostToDevice ) );
    else
        HANDLE_ERROR( cudaMemcpy( a, dev_a,size * sizeof( *dev_a ),cudaMemcpyDeviceToHost ) );
    }
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,start, stop ) );
    free( a );
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );
    return elapsedTime;
}

float cuda_host_alloc_test( int size, bool up ) {
    cudaEvent_t start, stop;int *a, *dev_a;
    float elapsedTime;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&a, size * sizeof( *a ),cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a,size * sizeof( *dev_a ) ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );
    for (int i=0; i<TOTAL; i++) {
        if (up)
        HANDLE_ERROR( cudaMemcpy( dev_a, a, size * sizeof( *a ),cudaMemcpyHostToDevice ) );
        else
        HANDLE_ERROR( cudaMemcpy( a, dev_a,size * sizeof( *a ),cudaMemcpyDeviceToHost ) );
    }
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,start, stop ) );
    HANDLE_ERROR( cudaFreeHost( a ) );
    HANDLE_ERROR( cudaFree( dev_a ) );
    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );
    return elapsedTime;
}

//#define SIZE    (10*1024*1024)
//#define SIZE    (1*1024)
//#define SIZE    (4*1024)
//#define SIZE    (16*1024)
//#define SIZE      (64*1024)
//#define SIZE    (1024*1024)
#define SIZE    (4*1024*1024)
//#define SIZE    (16*1024*1024)


int main( void ) {
    float elapsedTime;
    float MB = (float)TOTAL*SIZE*sizeof(int)/1024/1024;

    printf( "Pinned:\n" );
    elapsedTime = cuda_host_alloc_test( SIZE, true );
    printf( "Up Time using cudaHostAlloc:%3.1f ms\n",elapsedTime );
    printf( "\tTransfer %d Bytes;  MB/s during copy down:%3.1f\n", SIZE*4, MB /(elapsedTime/1000) );

    elapsedTime = cuda_host_alloc_test( SIZE, false );
    printf( "Down Time using cudaHostAlloc:%3.1f ms\n",elapsedTime );
    printf( "\tTransfer %d Bytes;  MB/s during copy down:%3.1f\n", SIZE*4, MB /(elapsedTime/1000) );

    printf("\n-------------------------------\n\n");

    printf( "Native:\n" );
    elapsedTime = cuda_malloc_test( SIZE, true );
    printf( "Up Time using cudaMalloc:%3.1f ms\n",elapsedTime );
    printf( "\tTransfer %d Bytes;  MB/s during copy down:%3.1f\n", SIZE*4, MB /(elapsedTime/1000) );
    
    elapsedTime = cuda_malloc_test( SIZE, false );
    printf( "Down Time using cudaMalloc:%3.1f ms\n",elapsedTime );
    printf( "\tTransfer %d	 Bytes;  MB/s during copy down:%3.1f\n", SIZE*4, MB /(elapsedTime/1000) );
	
}