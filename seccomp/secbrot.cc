/* Copyright 2014 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Example main() function for Brotli library. */

#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>

#include <ctime>
#include <string>
#include "version.h"
#include "../dec/decode.h"
#include "../dec/state.h"
#include "../enc/encode.h"
#include "Seccomp.hh"
#include "Zlib0.hh"

template <size_t buf_size> class BrotliFdIn : public brotli::BrotliIn {
    int fd_;
    uint8_t buf_[buf_size];
    bool eof_;
public:
    BrotliFdIn() {
        fd_ = -1;
        eof_ = false;
    }
    void init(int fd) {
        fd_ = fd;
    }
    bool is_eof() const {
        return eof_;
    }
    const void * Read(size_t n, size_t *bytes_read) {
        if (n == 0) {
            // If the caller passes n == 0 they are checking if EOF happened
            return eof_ ? NULL : buf_;
        }
        if (n > buf_size) {
            n = buf_size;
        }
        ssize_t ret;
        while ((ret = read(fd_, buf_, n)) < 0 && errno == EINTR) {

        }
        *bytes_read = ret;
        if (*bytes_read == 0) {
            eof_ = true;
            return NULL;
        }else {
            return buf_;
        }
    }
    void Close() {
        while(close(fd_) < 0 && errno == EINTR) {

        }
    }
};

class BrotliFdOut : public brotli::BrotliOut {
    int fd_;
public:
    BrotliFdOut() {
        fd_ = -1;
    }
    void init(int fd) {
        fd_ = fd;
    }
    bool Write(const void* buf, size_t n) {
        const unsigned char * to_write = (const unsigned char*)buf;
        while(n > 0) {
            ssize_t ret = write(fd_, to_write, n);
            if (ret < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            if (ret == 0) {
                return false;
            }
            n -= ret;
            to_write += ret;
        }
        return true;
    }
    void Close() {
        while(close(fd_) < 0 && errno == EINTR) {

        }
    }
};

#define kFileBufferSize (1<<16)
BrotliFdIn<kFileBufferSize> brotli_in;

BrotliFdOut brotli_out;

bool ParseUint64(const char* s, uint64_t* output) {
    errno = 0;
    *output = strtoll(s, NULL, 10);
    if (errno != 0) {
        return false;
    }
    return true;
}

bool ParseQuality(const char* s, int* quality) {
  if (s[0] >= '0' && s[0] <= '9') {
    *quality = s[0] - '0';
    if (s[1] >= '0' && s[1] <= '9') {
      *quality = *quality * 10 + s[1] - '0';
      return s[2] == 0;
    }
    return s[1] == 0;
  }
  return false;
}

static void ParseArgv(int argc, char **argv,
                      char **input_path,
                      char **output_path,
                      uint64_t* time_bound_ms,
                      int *force,
                      int *quality,
                      int *decompress,
                      int *repeat,
                      int *verbose,
                      int *lgwin,
                      int *enforce_jail,
                      uint64_t *memory_bound) {
  *force = 0;
  *input_path = 0;
  *output_path = 0;
  *repeat = 1;
  *verbose = 0;
  *lgwin = 22;
  *memory_bound = 1024 * 1024 * 384;
  *time_bound_ms = 0;
  {
    size_t argv0_len = strlen(argv[0]);
    *decompress =
        argv0_len >= 5 && strcmp(&argv[0][argv0_len - 5], "unbro") == 0;
  }
  for (int k = 1; k < argc; ++k) {
    if (!strcmp("--revision", argv[k])) {
        fprintf(stderr, "%s\n", revision);
        exit(0);
    }
    if (!strcmp("--force", argv[k]) ||
        !strcmp("-f", argv[k])) {
      if (*force != 0) {
        goto error;
      }
      *force = 1;
      continue;
    } else if (!strcmp("--decompress", argv[k]) ||
               !strcmp("--uncompress", argv[k]) ||
               !strcmp("-d", argv[k])) {
      *decompress = 1;
      continue;
    } else if (!strcmp("--jailed", argv[k]) ||
               !strcmp("-j", argv[k])) {
      *enforce_jail = 1;
      continue;
    } else if (!strcmp("--verbose", argv[k]) ||
               !strcmp("-v", argv[k])) {
      if (*verbose != 0) {
        goto error;
      }
      *verbose = 1;
      continue;
    }
    if (k < argc - 1) {
      if (!strcmp("--input", argv[k]) ||
          !strcmp("--in", argv[k]) ||
          !strcmp("-i", argv[k])) {
        if (*input_path != 0) {
          goto error;
        }
        *input_path = argv[k + 1];
        ++k;
        continue;
      } else if (!strcmp("--output", argv[k]) ||
                 !strcmp("--out", argv[k]) ||
                 !strcmp("-o", argv[k])) {
        if (*output_path != 0) {
          goto error;
        }
        *output_path = argv[k + 1];
        ++k;
        continue;
      } else if (!strcmp("--quality", argv[k]) ||
                 !strcmp("-q", argv[k])) {
        if (!ParseQuality(argv[k + 1], quality)) {
          goto error;
        }
        ++k;
        continue;
      } else if (!strcmp("--timeboundms", argv[k]) ||
                 !strcmp("-t", argv[k])) {
        if (!ParseUint64(argv[k + 1], time_bound_ms)) {
          goto error;
        }
        ++k;
        continue;
      } else if (!strcmp("--memorymb", argv[k]) ||
                 !strcmp("-t", argv[k])) {
        if (!ParseUint64(argv[k + 1], memory_bound)) {
          goto error;
        }
        *memory_bound *= 1024 * 1024;
        ++k;
        continue;
      } else if (!strcmp("--repeat", argv[k]) ||
                 !strcmp("-r", argv[k])) {
        if (!ParseQuality(argv[k + 1], repeat)) {
          goto error;
        }
        ++k;
        continue;
      }  else if (!strcmp("--window", argv[k]) ||
                  !strcmp("-w", argv[k])) {
        if (!ParseQuality(argv[k + 1], lgwin)) {
          goto error;
        }
        if (*lgwin < 10 || *lgwin >= 25) {
          goto error;
        }
        ++k;
        continue;
      }
    }
    goto error;
  }
  return;
error:
  fprintf(stderr,
          "Usage: %s [--force] [--quality n] [--decompress]"
          " [--input filename] [--output filename] [--repeat iters]"
          " [--verbose] [--window n]\n",
          argv[0]);
  exit(1);
}

static int OpenInputFile(const char* input_path) {
  if (input_path == 0) {
    return STDIN_FILENO;
  }
  int fd = open(input_path, O_RDONLY);
  if (fd < 0) {
    perror("open()");
    exit(1);
  }
  return fd;
}

static int OpenOutputFile(const char *output_path, const int force) {
  if (output_path == 0) {
    return STDOUT_FILENO;
  }
  int excl = force ? 0 : O_EXCL;
  int fd = open(output_path, O_CREAT | excl | O_WRONLY | O_TRUNC,
                S_IRUSR | S_IWUSR);
  if (fd < 0) {
    if (!force) {
      struct stat statbuf;
      if (stat(output_path, &statbuf) == 0) {
        fprintf(stderr, "output file exists\n");
        exit(1);
      }
    }
    perror("open");
    exit(1);
  }
  return fd;
}

int64_t FileSize(char *path) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    return -1;
  }
  if (fseek(f, 0L, SEEK_END) != 0) {
    fclose(f);
    return -1;
  }
  int64_t retval = ftell(f);
  if (fclose(f) != 0) {
    return -1;
  }
  return retval;
}

void *brotli_compat_custom_alloc(void *, size_t size) {
    return custom_malloc(size);
}
void brotli_compat_custom_free(void *, void*addr) {
    custom_free(addr);
}

bool Decompress(BrotliFdIn<kFileBufferSize>& fdin, brotli::BrotliOut &fdout) {
  unsigned char *output = new unsigned char[kFileBufferSize];
  size_t total_out;
  size_t available_in;
  const uint8_t* next_in;
  size_t available_out = kFileBufferSize;
  uint8_t* next_out = output;
  BrotliResult result = BROTLI_RESULT_NEEDS_MORE_INPUT;
  BrotliState s;
  BrotliStateInitWithCustomAllocators(&s, &brotli_compat_custom_alloc, &brotli_compat_custom_free,
                                      NULL);
  while (1) {
    if (result == BROTLI_RESULT_NEEDS_MORE_INPUT) {
      if (fdin.is_eof()) {
        break;
      }
      next_in = (const uint8_t*)fdin.Read(kFileBufferSize, &available_in);
      if (!next_in) {
          break;
      }
    } else if (result == BROTLI_RESULT_NEEDS_MORE_OUTPUT) {
      if (!fdout.Write(output, kFileBufferSize)) {
        break;
      }
      available_out = kFileBufferSize;
      next_out = output;
    } else {
      break; // Error or success.
    }
    result = BrotliDecompressStream(&available_in, &next_in,
        &available_out, &next_out, &total_out, &s);
  }
  if (next_out != output) {
    fdout.Write(output, next_out - output);
  }
  BrotliStateCleanup(&s);
  delete[] output;
  if (result == BROTLI_RESULT_NEEDS_MORE_OUTPUT) {
    fprintf(stderr, "failed to write output\n");
    exit(1);
  } else if (result != BROTLI_RESULT_SUCCESS) { // Error or needs more input.
    fprintf(stderr, "corrupt input\n");
    exit(1);
  }
  return true;
}
void print_memory_stats() {
    size_t memory = Sirikata::memmgr_size_allocated();
    char data[] = "XXXXXXXXXXXXXXXXXX bytes allocated\n";
    char * cursor = data + 18;
    for (int i = 0; i < 17; ++i) {
        --cursor;
        *cursor = '0' + memory % 10;
        memory /= 10;
        if (!memory) {
            break;
        }
    }
    while (write(2, cursor, strlen(cursor)) < 0 && errno == EINTR) {
    }
}

int main(int argc, char** argv) {
  char *input_path = 0;
  char *output_path = 0;
  int force = 0;
  int quality = 11;
  int decompress = 0;
  int repeat = 1;
  int verbose = 0;
  int lgwin = 0;
  size_t outputSize = 0;
  uint64_t time_bound_ms = 0;
  uint64_t memory_bound = 0;
  int enforce_jail = 0;
  ParseArgv(argc, argv, &input_path, &output_path, &time_bound_ms,
            &force, &quality, &decompress, &repeat, &verbose, &lgwin,
            &enforce_jail,
            &memory_bound);
  Sirikata::memmgr_init(memory_bound, 0, 0);
  const clock_t clock_start = clock();
  for (int i = 0; i < repeat; ++i) {
    brotli_in.init(OpenInputFile(input_path));
    brotli_out.init(OpenOutputFile(output_path, force));
    //setup timer
    if (time_bound_ms) {
        struct itimerval bound;
        bound.it_value.tv_sec = time_bound_ms / 1000;
        bound.it_value.tv_usec = (time_bound_ms % 1000) * 1000;
        bound.it_interval.tv_sec = 0;
        bound.it_interval.tv_usec = 0;
        int ret = setitimer(ITIMER_REAL, &bound, NULL);
        assert(ret == 0 && "Timer must be able to be set");
    }
    bool jailed = Sirikata::installStrictSyscallFilter(verbose);
    if (enforce_jail && !jailed) {
        custom_exit(ExitCode::JAIL_NOT_STARTED);
    }
    if (decompress) {
      Decompress(brotli_in, brotli_out);
      if (jailed) {
        if (verbose) {
          print_memory_stats();
        }
        custom_exit(ExitCode::SUCCESS);
      }
    } else {
      brotli::BrotliParams params;
      params.lgwin = lgwin;
      params.quality = quality;
      try {
        if (!BrotliCompress(params, &brotli_in, &brotli_out)) {
          fprintf(stderr, "compression failed\n");
          if (!jailed) {
              unlink(output_path);
          }
          custom_exit(ExitCode::ASSERTION_FAILURE);
        }
      } catch (std::bad_alloc&) {
        if(!jailed) {
          fprintf(stderr, "not enough memory\n");
          unlink(output_path);
        }
        custom_exit(ExitCode::OOM);
      }
    }
    if (verbose) {
        print_memory_stats();
    }
    if (jailed) {
        custom_exit(ExitCode::SUCCESS);
    }
    brotli_in.Close();
    brotli_out.Close();
  }
  if (verbose) {
    const clock_t clock_end = clock();
    double duration =
        static_cast<double>(clock_end - clock_start) / CLOCKS_PER_SEC;
    if (duration < 1e-9) {
      duration = 1e-9;
    }
    int64_t uncompressed_size = FileSize(decompress ? output_path : input_path);
    if (uncompressed_size != -1) {
      double uncompressed_bytes_in_MB =
        (repeat * uncompressed_size) / (1024.0 * 1024.0);
      if (decompress) {
        fprintf(stderr, "Brotli decompression speed: ");
      } else {
        fprintf(stderr, "Brotli compression speed: ");
      }
      fprintf(stderr, "%g MB/s\n", uncompressed_bytes_in_MB / duration);
    }
  }
  return 0;
}
