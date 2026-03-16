package main;

import (
	"crypto/rand"
	"fmt"
)

func main() {
	randomBytes := make([]byte, 32)
	rand.Read(randomBytes)
	fmt.Printf("%x\n", randomBytes)
}
