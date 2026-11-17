CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
LDLIBS ?= -lm

.PHONY: all run clean
all: xor

xor: minitf.c minitf.h examples/xor.c
	$(CC) $(CFLAGS) -I. minitf.c examples/xor.c -o xor $(LDLIBS)

run: xor
	./xor

clean:
	rm -f xor
