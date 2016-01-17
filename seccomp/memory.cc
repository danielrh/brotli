#include "memory.hh"
#include <unistd.h>
#include <errno.h>
#ifdef __linux
#include <sys/syscall.h>
#endif
#if defined(__APPLE__) || __cplusplus <= 199711L
#define thread_local __thread
#endif
extern "C" {
void* custom_malloc (size_t size) {
#ifdef USE_STANDARD_MEMORY_ALLOCATORS
    return malloc(size);
#else
    void * retval = Sirikata::memmgr_alloc(size);
    if (retval == 0) {// did malloc succeed?
        if (!g_use_seccomp) {
            assert(false && "Out of memory error");
        }
        custom_exit(ExitCode::OOM); // ran out of memory
    }
    return retval;
#endif
}

void* custom_realloc (void * old, size_t size) {
#ifdef USE_STANDARD_MEMORY_ALLOCATORS
    return realloc(old, size);
#else
    size_t actual_size = 0;
    void * retval = Sirikata::MemMgrAllocatorRealloc(old, size, &actual_size, true, NULL);
    if (retval == 0) {// did malloc succeed?
        if (!g_use_seccomp) {
            assert(false && "Out of memory error");
        }
        custom_exit(ExitCode::OOM); // ran out of memory
    }
    return retval;
#endif
}
void custom_free(void* ptr) {
#ifdef USE_STANDARD_MEMORY_ALLOCATORS
    free(ptr);
#else
    Sirikata::memmgr_free(ptr);
#endif
}

void * custom_calloc(size_t size, unsigned int count) {
#ifdef USE_STANDARD_MEMORY_ALLOCATORS
    return calloc(size, count);
#else
    size *= count;
    void * retval = Sirikata::memmgr_alloc(size); // guaranteed to return 0'd memory
    if (retval == 0) {// did malloc succeed?
        if (!g_use_seccomp) {
            assert(false && "Out of memory error");
        }
        custom_exit(ExitCode::OOM); // ran out of memory
    }
    return retval;
#endif
}
}
bool g_use_seccomp =
#ifndef __linux
    false
#else
    true
#endif
    ;
void * operator new(size_t size, std::nothrow_t const&) {
 void* ptr = custom_malloc(size); 
 if (ptr == 0) {// did malloc succeed?
     if (!g_use_seccomp) {
         assert(false && "Out of memory error");
     }
     custom_exit(ExitCode::OOM); // ran out of memory
 }
 return ptr;
}
void* operator new (size_t size) throw(std::bad_alloc){
 void* ptr = custom_malloc(size); 
 if (ptr == 0) {// did malloc succeed?
     if (!g_use_seccomp) {
         assert(false && "Out of memory error");
     }
     custom_exit(ExitCode::OOM); // ran out of memory
 }
 return ptr;
}

void* operator new[] (size_t size) throw(std::bad_alloc){
 void* ptr = custom_malloc(size); 
 if (ptr == 0) {// did malloc succeed?
     if (!g_use_seccomp) {
         assert(false && "Out of memory error");
     }
     custom_exit(ExitCode::OOM); // ran out of memory
 }
 return ptr;
}

void operator delete(void * ptr, std::nothrow_t const&) {
    custom_free(ptr);
}
void operator delete[](void * ptr, std::nothrow_t const&) {
    custom_free(ptr);
}
void operator delete (void* ptr) throw(){
    custom_free(ptr);
}
void operator delete[] (void* ptr) throw(){
    custom_free(ptr);
}
thread_local int l_emergency_close_signal = -1;
thread_local void (*atexit_f)(void*) = NULL;
thread_local void *atexit_arg = NULL;
void custom_atexit(void (*atexit)(void*) , void *arg) {
    assert(!atexit_f);
    atexit_f = atexit;
    atexit_arg = arg;
}

void custom_exit(ExitCode::ExitCode exit_code) {
    if (atexit_f) {
        (*atexit_f)(atexit_arg);
        atexit_f = NULL;
    }
#ifdef __linux
    syscall(SYS_exit, (int)exit_code);
#else
    exit((int)exit_code);
#endif
    abort();
}
