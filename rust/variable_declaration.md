# Understanding `let` vs `let mut` in Rust

**Question**: How `let` and `let mut` are represented in the memory? How does the internal representation of the two variables and how much do they differ?

**Abstract**: A Deep Dive into Compiler Internals and Memory Representation in Rust.

**Binding as Semantic Guarantee**

In Rust, mutability is a property of bindings that is determined at compile time, unlike imperative languages where variables can be reassigned freely. When you declare a variable with `let x = 5;`, it creates an immutable binding, similar to a single-assignment construct, where the identifier `x` is associated with the value 5 for its entire scope. If you use `let mut x = 5;`, you can reassign the value of `x` later. The Rust compiler checks that you follow this rule through type checking and borrow analysis, and it will not generate code if you try to break it, as stated by the Rust Language Team in 2024.

**Mid-Level Intermediate Representation (MIR) Analysis**

Rust's compiler converts source code into [MIR](https://blog.rust-lang.org/2016/04/19/MIR/), a structured control-flow graph that can be verified for safety. When MIR encounters an immutable binding, it assigns the value to a local slot that is marked as frozen after initialization. If the code tries to assign a new value to this slot with a `Statement::Assign` operation, the borrow checker will throw an error. This is not the case for mutable bindings, which can be assigned multiple times. The compiler checks for invalid writes, such as trying to assign a new value to an immutable variable, and stops compilation if it finds any. This prevents the need for runtime checks to enforce these rules.

**LLVM Intermediate Representation and Optimization**

MIR is translated to LLVM IR, and mutability affects how code is generated and optimized. When bindings are immutable, the compiler can perform aggressive transformations. For example, constant propagation replaces variable references with literals, and register allocation eliminates memory accesses because the compiler can prove that the values do not change. In contrast, mutable bindings require more conservative approaches. The compiler uses allocation instructions to allocate writable stack slots, and it uses load and store pairs, such as `load i32* %x, store i32 %val, i32* %x`, to preserve the values for potential modifications. 

This also applies to shadowing, where a new identifier is bound to a different value, as in 

```rust
let x = 5; 
let x = x + 1;
```
 
which creates a new slot without aliasing the original identifier.

Shadowing is a concept that needs to be understood. You can learn more about shadowing at [here](https://doc.rust-lang.org/rust-by-example/variable_bindings/scope.html).



**Runtime Stack Mechanics and Hardware Abstraction**

At the machine level, both binding types use stack-frame offsets. The CPU executes the same load and store instructions, such as x86 `mov`, and cannot distinguish between them. Immutability is enforced by the compiler's type system, which prevents store generation for immutable targets and makes writes impossible. This approach is consistent with Rust's memory safety paradigm, which relies on ownership and borrowing to prevent errors at runtime.

**Theoretical Foundations and Implications**

Rust's design uses a type-theoretic single-assignment discipline for immutable bindings, which supports functional programming and optimization. Mutable bindings, on the other hand, allow for changes to state. This approach to mutability helps to prevent aliasing hazards, making it easier to reason about code, and allows for efficient abstractions. The LLVM compiler takes advantage of Rust's immutable bindings, using this information to perform optimizations like dead store elimination.

**Practical Implementation**

Consider these two nearly identical functions:

```rust
pub fn immutable_example() -> i32 {
    let x = 10;
    x
}

pub fn mutable_example() -> i32 {
    let mut x = 10;
    x = 20;
    x
}
```

The first function creates an immutable binding and returns its value. The second creates a mutable binding, modifies it, and returns the new value.

**Examining the Assembly Output**

Using [Compiler Explorer](https://rust.godbolt.org/), we can examine the generated assembly. For the immutable version, the compiler often realizes that `x` is always 10 and generates simply:

```text
example::mutable_example::he8a677873798bbd4:
	mov dword ptr [rsp - 4], 10
	mov dword ptr [rsp - 4], 20
	mov eax, dword ptr [rsp - 4]
	ret

example::immutable_example::hf3c9ab22a22969b2:
	mov dword ptr [rsp - 4], 10
	mov eax, 10
	ret
```

In the mutable example, the assembly shows the hardware executing the "Read-Write" permission granted by the compiler. The instruction `mov dword ptr [rsp - 4], 10` sets the initial value at a specific stack offset. The next line, `mov dword ptr [rsp - 4], 20`, overwrites this value. This shows that for `let mut`, the compiler generates code that treats the stack slot as a dynamic workspace. The final value is loaded into the `eax` register for the return.

The **immutable example** shows how the `let` keyword acts as a "gatekeeper". The compiler initializes the memory at `[rsp - 4]` with 10, probably for safety or debugging symbols. Because the binding is immutable and will never change, the optimizer can perform **constant folding**. It hardcodes `mov eax, 10` instead of reading from the stack address. The compiler bypasses the "read" from memory because it knows the value at that address is a constant.

**The MIR Output**

```text
rust_mir
// WARNING: This output format is intended for human consumers only
// and is subject to change without notice. Knock yourself out.
// HINT: See also -Z dump-mir for MIR at specific points during compilation.
fn immutable_example() -> i32 {
    let mut _0: i32;
    scope 1 {
        debug x => const 10_i32;
    }

    bb0: {
        _0 = const 10_i32;
        return;
    }
}

fn mutable_example() -> i32 {
    let mut _0: i32;
    let mut _1: i32;
    scope 1 {
        debug x => _1;
    }

    bb0: {
        _1 = const 10_i32;
        _1 = const 20_i32;
        _0 = copy _1;
        return;
    }
}
```

In the `immutable_example`, the compiler does not track `x` as a variable that needs a memory slot. It treats `x` as a debug alias for the constant `10_i32`. The compiler simplifies the MIR to a single assignment to the return value, `_0`, because there is no `mut`. Since `x` is defined as a constant in the debug scope, the compiler cannot generate a second assignment instruction for it. 

The MIR architecture is different in the `mutable_example`. The debug scope points to a dedicated storage slot, `_1`, instead of a constant. Within the basic block, `bb0`, there are sequential write instructions: first assigning `10_i32` and then overwriting it with `20_i32`. This confirms your intuition. The `let mut` keyword forces the compiler to create a live variable slot, `_1`, that can undergo multiple state changes. In contrast, the standard `let` allows the compiler to collapse the variable into a static value.

To see the MIR output you can use the following command in `rustc`:

```sh
rustc --emit=mir <filename.rs>
```

*Explore point*: Borrow Checker errors on writing to the immutable function.

**Visual Memory Representation**

For an immutable binding `let x = 10`:
- **Stack Slot**: Memory address at offset +0
- **Value**: 10 (stored as bytes)
- **Metadata**: Read-only flag set in compiler's internal representation
- **Operations permitted**: Load
- **Operations rejected**: Store to this location

For a mutable binding `let mut x = 10`:
- **Stack Slot**: Memory address at offset +0
- **Value**: 10 (initial), may change to 20, 30, etc.
- **Metadata**: Read-write flag set in compiler's internal representation
- **Operations permitted**: Load and Store

At last:
- The hardware does not differentiate between mutable and immutable memory, it just executes read and write instructions to addresses. This helps in Rust design for the program optimization.

**Reference**  
[1] Rust Language Team. (2024). _The Rust Reference: Variables_. Retrieved from: https://doc.rust-lang.org/reference/variables.html
[2] Rust by Example. *Variable Binding: Scope and Shadowing*. Retrieved from: https://doc.rust-lang.org/rust-by-example/variable_bindings/scope.html 
