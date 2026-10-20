package ccengine

import (
	"sort"
	"sync"
	"time"
)

type Summary struct {
	Strategy                               string
	Total, Committed, Wasted, Locked       int
	AbortRate, ThroughputTPS, P50ms, P99ms float64
}

// Run executes txns concurrently on a fresh store and aggregates metrics.
func Run(strategy CCStrategy, keys []string, txns []Transaction, workers int) Summary {
	store := NewStoreFromKeys(keys)
	jobs := make(chan int)
	results := make([]Result, len(txns))
	var wg sync.WaitGroup
	start := time.Now()
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				results[i] = strategy.Execute(store, &txns[i])
			}
		}()
	}
	for i := range txns {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	elapsed := time.Since(start).Seconds()

	s := Summary{Strategy: strategy.Name(), Total: len(txns)}
	lat := make([]float64, 0, len(results))
	for _, r := range results {
		if r.Committed {
			s.Committed++
		}
		if r.WastedWork() {
			s.Wasted++
		}
		if r.UsedLocking {
			s.Locked++
		}
		lat = append(lat, float64(r.LatencyNs)/1e6)
	}
	sort.Float64s(lat)
	s.AbortRate = float64(s.Wasted) / float64(s.Total)
	s.ThroughputTPS = float64(s.Committed) / elapsed
	s.P50ms, s.P99ms = lat[len(lat)/2], lat[int(float64(len(lat))*0.99)]
	return s
}
