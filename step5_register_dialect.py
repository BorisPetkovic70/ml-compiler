from xdsl.context import Context
from xdsl.ir import Dialect
from xdsl.dialects.builtin import Builtin
from xdsl.dialects import func


# For now: Empty HiCompiler dialect (no ops / attributes yet)
HiCompiler = Dialect(
    "hc",      # dialect name
    (),        # operations
    (),        # attributes / types
)


def main() -> None:
    ctx = Context()

    # Load the built-in dialects you already use
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)

    # ✅ Now load your custom HiCompiler dialect instance
    ctx.load_dialect(HiCompiler)

    # The point is just: “can we register & load the dialect without errors?”
    print(ctx)
    print("\nHiCompiler dialect registered and loaded successfully.")


if __name__ == "__main__":
    main()

