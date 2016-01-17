#brotli/tools

include ../shared.mk

BROTLI = ..
ENCOBJ = $(BROTLI)/enc/*.o
DECOBJ = $(BROTLI)/dec/*.o

EXECUTABLES=secbrot

EXE_OBJS=memory.o Zlib0.o MemMgrAllocator.o Seccomp.o $(patsubst %, %.o, $(EXECUTABLES))

all : $(EXECUTABLES)

$(EXECUTABLES) : $(EXE_OBJS) deps
	$(CXX) $(LDFLAGS) $(EXE_OBJS) $(ENCOBJ) $(DECOBJ) -o $@ version.cpp -DGIT_REVISION=\"$(shell git rev-parse HEAD)\"

deps :
	$(MAKE) -C $(BROTLI)/dec
	$(MAKE) -C $(BROTLI)/enc nodict

clean :
	rm -f $(OBJS) $(EXE_OBJS) $(EXECUTABLES)
	$(MAKE) -C $(BROTLI)/dec clean
	$(MAKE) -C $(BROTLI)/enc clean
