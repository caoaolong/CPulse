# ============================================================================
#  Makefile - CPulse: C inference engine for Qwen2.5-0.5B
# ============================================================================
#  Usage (Cygwin on this machine: D:/cygwin64/bin/make.exe):
#    make                build release binary  -> build/cpulse.exe
#    make debug          build debug binary    -> build/debug/cpulse.exe
#    make run            build release, then run it from the project root
#    make clean          remove the entire build/ directory
#
#  Notes:
#    - Platform is auto-detected via `uname -s`. Under Cygwin / MSYS2 /
#      MinGW / Linux / macOS the recipe shell already understands
#      `mkdir -p` / `rm -rf`, so plain POSIX commands are used.
#      Only a native Windows port of make (e.g. mingw32-make, where
#      `uname` is unavailable) falls back to `cmd /c` builtins.
#      (Do NOT test `$(OS)` alone: Windows exports OS=Windows_NT even
#      inside Cygwin shells, so it cannot tell Cygwin apart from
#      native Windows make.)
#    - The `run` target must be executed from the project root, because the
#      default model path is a relative path: .cache/Qwen2.5-0.5B
# ============================================================================

# ---------------------------------------------------------------- toolchain
CC      := gcc
BUILD   := build
INC     := include
SRC     := src
TARGET_NAME := cpulse.exe

# ---------------------------------------------------------------- platform
# `2>&1` (no /dev/null redirect) keeps this line valid in both POSIX sh
# and cmd.exe. When uname is missing (native Windows make) its "not
# recognized" error simply matches none of the patterns below.
UNAME_S := $(shell uname -s 2>&1)
ifneq ($(findstring CYGWIN,$(UNAME_S)),)
  IS_UNIX_SHELL := 1
else ifneq ($(findstring MSYS,$(UNAME_S)),)
  IS_UNIX_SHELL := 1
else ifneq ($(findstring MINGW,$(UNAME_S)),)
  IS_UNIX_SHELL := 1
else ifneq ($(findstring Linux,$(UNAME_S)),)
  IS_UNIX_SHELL := 1
else ifneq ($(findstring Darwin,$(UNAME_S)),)
  IS_UNIX_SHELL := 1
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

# ---------------------------------------------------------------- mkdir / rm / run
# Cygwin/MSYS2/MinGW/Linux/macOS: POSIX shell -> plain mkdir -p / rm -rf,
# binaries run as ./path.
# Native Windows make (mingw32-make, uname unavailable): route directory
# operations through cmd /c builtins (native make executes simple recipe
# lines via CreateProcess, and Windows has no mkdir.exe / rm.exe), and run
# binaries with backslash paths.
#
# NOTE: do NOT quote the paths in the cmd branch. Native make mangles quoted
# command lines and cmd.exe then silently falls back to interactive mode
# (nothing gets executed). Paths are converted to backslash form for cmd.exe.
ifdef IS_UNIX_SHELL
  MKDIR_P = mkdir -p $(1)
  RM_RF   = rm -rf $(1)
  RUN_BIN = ./$(1)
else ifeq ($(OS),Windows_NT)
  MKDIR_P = cmd /c if not exist $(subst /,\,$(1)) mkdir $(subst /,\,$(1))
  RM_RF   = cmd /c rmdir /s /q $(subst /,\,$(1))
  RUN_BIN = $(subst /,\,$(1))
else
  MKDIR_P = mkdir -p $(1)
  RM_RF   = rm -rf $(1)
  RUN_BIN = ./$(1)
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
	$(call RUN_BIN,$(BUILD)/$(TARGET_NAME))

clean:
	$(call RM_RF,$(BUILD))

help:
	@echo "targets: all (default) | debug | run | clean"

# auto-include generated dependency files (incremental rebuild correctness)
-include $(DEPS)
