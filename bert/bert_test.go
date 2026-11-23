package bert_test

import (
	"slices"
	"testing"
	"testing/fstest"

	"github.com/pkieltyka/goinfer/bert"
)

const wordPiece = `{
  "version":"1.0",
  "added_tokens":[{"id":2,"content":"[CLS]","special":true},{"id":3,"content":"[SEP]","special":true}],
  "normalizer":{"type":"BertNormalizer","lowercase":true},
  "pre_tokenizer":{"type":"BertPreTokenizer"},
  "model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,
           "vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"hello":4,"world":5,"##s":6}}
}`

func TestWordPieceFromFS(t *testing.T) {
	files := fstest.MapFS{"model/tokenizer.json": {Data: []byte(wordPiece)}}
	tok, err := bert.LoadTokenizerFromFS(files, "model/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Encode("Hello Worlds"); !slices.Equal(got, []int32{4, 5, 6}) {
		t.Errorf("Encode = %v, want [4 5 6]", got)
	}
	for literal, want := range map[string]int32{"[CLS]": 2, "[SEP]": 3} {
		if id, ok := tok.SpecialID(literal); !ok || id != want {
			t.Errorf("SpecialID(%q) = %d, %v, want %d, true", literal, id, ok, want)
		}
	}
	if tok, err := bert.LoadTokenizerFromFS(files, "missing.json"); err == nil || tok != nil {
		t.Fatalf("missing tokenizer returned %v, %v", tok, err)
	}
}

func TestBERTZeroAndMissingFS(t *testing.T) {
	var empty bert.BERT
	if empty.Tokenizer() != nil {
		t.Fatal("zero model returned a tokenizer")
	}
	if err := empty.Close(); err != nil {
		t.Fatal(err)
	}
	if model, err := bert.LoadBERTFromFS(fstest.MapFS{}, "model"); err == nil || model != nil {
		t.Fatalf("missing model returned %v, %v", model, err)
	}
}
