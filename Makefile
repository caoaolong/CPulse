# ============================================================================
#  Makefile - CPulse: C inference engine for Qwen2.5-0.5B
# ============================================================================
#  Usage (on Windows with MSYS2/mingw32-make):
#    mingw32-make            build release binary  -> build/cpulse.exe
#    mingw32-make debug      build debug binary    -> build/debug/cpulse.exe
#    mingw32-make run        build release, then run it from the project root
#    mingw32-make clean      remove the entire build/ directory
#
#  Notes:
#    - mingw32-make on Windows defaults to cmd.exe, which cannot run
#      `mkdir -p` / `rm -rf`. We therefore explicitly select MSYS2's bash
#      as the recipe shell. If your MSYS2 lives elsewhere, adjust SHELL below.
#    - The `run` target must be executed from the project root, because the
#      default model path is a relative path: .cache/Qwen2.5-0.5B
# ============================================================================

# ---------------------------------------------------------------- toolchain
CC      := gcc
BUILD   := build
INC     := include
SRC     := src
TARGET_NAME := cpulse.exe

# Cross-platform recipe shell.
# Linux/macOS: system shell already handles mkdir -p / rm -rf.
ifeq ($(OS),Windows_NT)
  SHELL       := C:/msys64/usr/bin/bash.exe
  .SHELLFLAGS := -c
endif

# ---------------------------------------------------------------- sources
SRCS := $(wildcard $(SRC)/*.c)
OBJS := $(patsubst $(SRC)/%.c,$(BUILD)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

# ---------------------------------------------------------------- flags
CFLAGS_RELEASE := -O2 -std=c11 -Wall -Wextra -I$(INC)
CFLAGS_DEBUG   := -O0 -g3 -std=c11 -Wall -Wextra -I$(INC)
LDLIBS         := -lm

# default (no target given): release
CFLAGS ?= $(CFLAGS_RELEASE)

# ---------------------------------------------------------------- mkdir / rm
# mingw32-make (native Windows port) executes "simple" recipe lines directly
# via CreateProcess, bypassing the SHELL variable. Since Windows has no
# mkdir.exe / rm.exe, we route directory operations through cmd /c builtins.
#
# NOTE: do NOT quote the paths here. Native make mangles quoted command
# lines and cmd.exe then silently falls back to interactive mode (nothing
# gets executed). Paths are converted to backslash form for cmd.exe.
ifeq ($(OS),Windows_NT)
  MKDIR_P = cmd /c if not exist $(subst /,\,$(1)) mkdir $(subst /,\,$(1))
  RM_RF   = cmd /c rmdir /s /q $(subst /,\,$(1))
else
  MKDIR_P = mkdir -p $(1)
  RM_RF   = rm -rf $(1)
endif

# ---------------------------------------------------------------- targets
.PHONY: all debug run clean help

all: $(BUILD)/$(TARGET_NAME)

# create build directory (order-only prerequisite: does not trigger rebuilds)
$(BUILD):
	$(call MKDIR_P,$(BUILD))

# compile: .c -> .o (also emit dependency files for incremental builds)
$(BUILD)/%.o: $(SRC)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# link
$(BUILD)/$(TARGET_NAME): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDLIBS)

# debug build: separate output dir so it never overwrites the release binary
debug:
	$(MAKE) BUILD=$(BUILD)/debug CFLAGS="$(CFLAGS_DEBUG)" all

# run from project root (model path is relative to the root)
run: all
	./$(BUILD)/$(TARGET_NAME)

clean:
	$(call RM_RF,$(BUILD))

help:
	@echo "targets: all (default) | debug | run | clean"

# auto-include generated dependency files (incremental rebuild correctness)
-include $(DEPS)
