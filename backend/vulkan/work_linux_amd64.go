package vulkan

import (
	"fmt"

	"github.com/pkieltyka/goinfer/backend/vulkan/internal/vk"
)

// A submission admits the padded FMA work of 128 rows of the real PLE
// projection. This bounds submitted work, independent of device speed.
const submissionWork = 128 * 512 * 12288

const queryBlock = 128

type workGroup struct {
	units      uint64
	dispatches int
	commands   []func(*vk.Commands) error
}

type workPlan struct {
	groups []workGroup
	err    error
	budget int
}

func (p *workPlan) add(name string, units uint64, dispatch bool, command func(*vk.Commands) error) {
	if p.err != nil {
		return
	}
	if units > submissionWork || p.budget < 1 {
		p.err = fmt.Errorf("vulkan: %s requires %d work units with a %d-unit submission limit and %d descriptor sets", name, units, submissionWork, p.budget)
		return
	}
	if len(p.groups) == 0 {
		p.groups = append(p.groups, workGroup{})
	}
	group := &p.groups[len(p.groups)-1]
	if group.units+units > submissionWork || (dispatch && group.dispatches >= p.budget) {
		p.groups = append(p.groups, workGroup{})
		group = &p.groups[len(p.groups)-1]
	}
	group.units += units
	if dispatch {
		group.dispatches++
	}
	group.commands = append(group.commands, command)
}

func product(dimensions ...int) uint64 {
	n := uint64(1)
	for _, d := range dimensions {
		n *= uint64(d)
	}
	return n
}

func tile(n, size int) int { return ceilDiv(n, size) * size }

type attentionBlock struct {
	sequence, query, rows, key, keys int
}

func (b attentionBlock) scores(heads, dim int) uint64 {
	return product(heads, tile(b.rows, 8), tile(b.keys, 16), tile(dim, 16))
}

func (b attentionBlock) values(heads, dim int) uint64 {
	return product(heads, tile(b.rows, 8), tile(dim, 16), tile(b.keys, 16))
}

func (b attentionBlock) softmax(heads int) uint64 {
	// Three trips across the compact tile, plus two shared reductions.
	return product(heads, b.rows, 3*tile(b.keys, 64)+128)
}

// Each block belongs to one sequence. The keys cover the union of its row
// windows. Shaders further clip each row at that row's exact window.
func attentionBlocks(lengths []int, heads, dim, radius int, visit func(attentionBlock)) error {
	sequence := 0
	for _, n := range lengths {
		for q := 0; q < n; {
			rows := min(queryBlock, n-q)
			first := max(0, q-radius)
			for {
				end := min(n, q+rows+radius)
				b := attentionBlock{sequence, sequence + q, rows, sequence + first, end - first}
				if b.scores(heads, dim) <= submissionWork && b.values(heads, dim) <= submissionWork && b.softmax(heads) <= submissionWork {
					visit(b)
					break
				}
				if rows == 1 {
					return fmt.Errorf("vulkan: one attention row exceeds the submission work limit")
				}
				rows /= 2
			}
			q += rows
		}
		sequence += n
	}
	return nil
}
