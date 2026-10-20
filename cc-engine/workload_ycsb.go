package ccengine

// YCSB-style generator, parameterised to reproduce the real Yahoo! Cloud
// Serving Benchmark core workloads A-F exactly as shipped in YCSB 0.17.0's
// workloads/ directory (read/update/insert proportions and request
// distribution below are copied verbatim from those files):
//
//   A  update-heavy     read=0.50 update=0.50            zipfian
//   B  read-mostly      read=0.95 update=0.05            zipfian
//   C  read-only        read=1.00                        zipfian
//   D  read-latest      read=0.95 insert=0.05             latest
//   F  read-modify-write read=0.50 rmw=0.50               zipfian
//
// Every core-workload operation in real YCSB touches exactly one record;
// OpsPerTxn>1 batches several such single-key operations into one
// multi-key SAPCC transaction, which is NOT part of the stock YCSB
// definition but is useful here to exercise the engine's multi-key
// validation/locking path. OpsPerTxn=1 reproduces stock YCSB exactly.

import (
	"fmt"
	"math/rand"
)

type YCSBWorkload string

const (
	YCSB_A YCSBWorkload = "A"
	YCSB_B YCSBWorkload = "B"
	YCSB_C YCSBWorkload = "C"
	YCSB_D YCSBWorkload = "D"
	YCSB_F YCSBWorkload = "F"
)

type ycsbMix struct {
	read, update, insert, rmw float64
	dist                      string
}

var ycsbMixes = map[YCSBWorkload]ycsbMix{
	YCSB_A: {read: 0.50, update: 0.50, dist: "zipfian"},
	YCSB_B: {read: 0.95, update: 0.05, dist: "zipfian"},
	YCSB_C: {read: 1.00, dist: "zipfian"},
	YCSB_D: {read: 0.95, insert: 0.05, dist: "latest"},
	YCSB_F: {read: 0.50, rmw: 0.50, dist: "zipfian"},
}

type YCSBConfig struct {
	Workload       YCSBWorkload
	RecordCount    int     // YCSB `recordcount`
	OperationCount int     // YCSB `operationcount`
	OpsPerTxn      int     // 1 = stock YCSB; >1 = batched extension (see above)
	ZipfS          float64 // Zipfian skew (YCSB's default ZipfianGenerator ~1.0 is almost uniform; this engine uses a steeper default to make contention visible over a small key space, matching how YCSB is actually re-tuned for CC benchmarking, e.g. by ForeSight/Bao-style evaluations)
	Seed           int64
}

func GenerateYCSB(c YCSBConfig) []Transaction {
	mix, ok := ycsbMixes[c.Workload]
	if !ok {
		panic(fmt.Sprintf("unknown YCSB workload %q", c.Workload))
	}
	if c.OpsPerTxn < 1 {
		c.OpsPerTxn = 1
	}
	r := rand.New(rand.NewSource(c.Seed))
	var zipf *rand.Zipf
	if mix.dist == "zipfian" {
		zipf = rand.NewZipf(r, c.ZipfS, 1, uint64(c.RecordCount-1))
	}

	// "latest" distribution (workload D): recently inserted records are hot.
	// YCSB implements this by Zipfian-weighting the offset back from the
	// most-recently-inserted key. insertedSoFar grows as D's 5% inserts run,
	// modelling exactly that recency bias without needing YCSB's Java runtime.
	insertedSoFar := c.RecordCount
	latestZipf := rand.NewZipf(r, 1.4, 1, uint64(c.RecordCount-1))

	pickKey := func() int {
		switch mix.dist {
		case "latest":
			back := int(latestZipf.Uint64())
			k := insertedSoFar - 1 - back
			if k < 0 {
				k = 0
			}
			return k
		default:
			return int(zipf.Uint64())
		}
	}

	txns := make([]Transaction, 0, c.OperationCount)
	for i := 0; i < c.OperationCount; {
		keys := make([]string, 0, c.OpsPerTxn)
		writes := make([]bool, 0, c.OpsPerTxn)
		seen := map[int]bool{}
		op := "READ"
		for j := 0; j < c.OpsPerTxn && i < c.OperationCount; j++ {
			x := r.Float64()
			var isWrite bool
			switch {
			case x < mix.read:
				op, isWrite = "READ", false
			case x < mix.read+mix.update:
				op, isWrite = "UPDATE", true
			case x < mix.read+mix.update+mix.insert:
				op, isWrite = "INSERT", true
				insertedSoFar++
			default:
				op, isWrite = "READ_MODIFY_WRITE", true
			}
			k := pickKey()
			if k >= c.RecordCount {
				k = c.RecordCount - 1
			}
			// A transaction must never touch the same key twice: lockSorted
			// takes each key's mutex once per transaction, and a repeat key
			// would self-deadlock the goroutine (Go mutexes aren't reentrant).
			if seen[k] {
				continue
			}
			seen[k] = true
			keys = append(keys, KeyName(k))
			writes = append(writes, isWrite)
			i++
		}
		if len(keys) == 0 {
			continue
		}
		txns = append(txns, Transaction{
			ID:   fmt.Sprintf("ycsb%s-%d", c.Workload, len(txns)),
			Keys: keys, Writes: writes,
			SemanticType: "YCSB_" + string(c.Workload) + "_" + op,
			SemanticText: fmt.Sprintf("YCSB workload %s %s operation touching %d key(s)", c.Workload, op, len(keys)),
		})
	}
	return txns
}
