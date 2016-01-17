OS := $(shell uname)

CC ?= gcc
CXX ?= g++

COMMON_FLAGS = -static -fno-omit-frame-pointer -no-canonical-prefixes -O2 -g -Doverridable_malloc=custom_malloc -include ../seccomp/memory.hh -Doverridable_realloc=custom_realloc -Doverridable_free=custom_free -Doverridable_calloc=custom_calloc
ifeq ($(OS), Darwin)
  CPPFLAGS += -DOS_MACOSX
endif

CFLAGS += $(COMMON_FLAGS) -Wmissing-prototypes
CXXFLAGS += $(COMMON_FLAGS) -Wmissing-declarations
