//go:build cuda

// Package ptxtest parses CUDA and PTX signatures for the backend's asset census tests.
package ptxtest

import (
	"regexp"
	"strconv"
	"strings"
)

var (
	ptxTarget    = regexp.MustCompile(`(?m)^\s*\.target\s+sm_([0-9]+)(?:[ ,\t]|$)`)
	ptxEntryHead = regexp.MustCompile(`\.visible\s+\.entry\s+([A-Za-z_][A-Za-z0-9_$]*)\s*\(`)
	ptxParamKind = regexp.MustCompile(`\.param\s+(?:\.align\s+\d+\s+)?\.([a-z]+[0-9]*)`)
	cuGlobalHead = regexp.MustCompile(`__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?void\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(`)
	comments     = regexp.MustCompile(`(?s)/\*.*?\*/|//[^\n]*`)
)

// TargetSM returns the numeric architecture declared by a PTX module.
func TargetSM(ptx []byte) (int, bool) {
	match := ptxTarget.FindSubmatch(ptx)
	if len(match) != 2 {
		return 0, false
	}
	sm, err := strconv.Atoi(string(match[1]))
	return sm, err == nil
}

// Entries returns the argument kinds of each visible PTX entry.
func Entries(ptx string) map[string][]string {
	out := map[string][]string{}
	for _, loc := range ptxEntryHead.FindAllStringSubmatchIndex(ptx, -1) {
		name := ptx[loc[2]:loc[3]]
		end := strings.Index(ptx[loc[1]:], ")")
		if end < 0 {
			continue
		}
		var kinds []string
		for _, pm := range ptxParamKind.FindAllStringSubmatch(ptx[loc[1]:loc[1]+end], -1) {
			kinds = append(kinds, ptxKind(pm[1]))
		}
		out[name] = kinds
	}
	return out
}

func ptxKind(t string) string {
	switch t {
	case "u64", "s64", "b64":
		return "64"
	case "u32", "s32", "b32":
		return "32i"
	case "f32":
		return "f32"
	case "f64":
		return "f64"
	case "u16", "s16", "b16", "f16":
		return "16"
	case "u8", "s8", "b8":
		return "8"
	}
	return "?"
}

// Globals returns the argument kinds of each CUDA global function.
func Globals(cu string) map[string][]string {
	src := comments.ReplaceAllString(cu, " ")
	out := map[string][]string{}
	for _, loc := range cuGlobalHead.FindAllStringSubmatchIndex(src, -1) {
		name := src[loc[2]:loc[3]]
		depth, i := 1, loc[1]
		for ; i < len(src) && depth > 0; i++ {
			switch src[i] {
			case '(':
				depth++
			case ')':
				depth--
			}
		}
		params := strings.TrimSpace(src[loc[1] : i-1])
		var kinds []string
		if params != "" && params != "void" {
			for p := range strings.SplitSeq(params, ",") {
				kinds = append(kinds, cudaKind(p))
			}
		}
		out[name] = kinds
	}
	return out
}

func cudaKind(p string) string {
	if strings.Contains(p, "*") {
		return "64"
	}
	f := strings.Fields(strings.NewReplacer("const", " ", "__restrict__", " ").Replace(p))
	if len(f) < 2 {
		return "?"
	}
	switch strings.Join(f[:len(f)-1], " ") {
	case "float":
		return "f32"
	case "double":
		return "f64"
	case "int", "unsigned", "unsigned int", "uint32_t", "int32_t", "uint":
		return "32i"
	case "long", "long long", "int64_t", "uint64_t", "size_t", "unsigned long long":
		return "64"
	case "__half", "half", "short", "unsigned short":
		return "16"
	case "bool", "char", "unsigned char", "int8_t", "uint8_t", "signed char":
		return "8"
	}
	return "?"
}
