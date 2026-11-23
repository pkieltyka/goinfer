# Frozen compatibility baseline

These are type exports for `chat`, `constrain`, `decoder`, and `tokenizer` from goinfer v0.1.0 (`bdf9daf95cae55e3ea79df61ba056f2ddf607947`). They capture the historical API without requiring the old release's aikit dependency when CI runs.

Produced from that exact tagged tree with `GOWORK=off`, `CGO_ENABLED=0`, Go 1.27.4 on Linux amd64 and `golang.org/x/exp/cmd/apidiff` at `v0.0.0-20260727155853-b88d891fe743`. For each package, the capture command was `apidiff -w "$output/$package.api" "github.com/pkieltyka/goinfer/$package"` from the historical source root. `SHA256SUMS` identifies the captured exports. The historical tree and the signed-off Hard-tier list determine the compatibility baseline.
