# PointChess Opening Book System

PointChess supports standard Polyglot binary opening books (`book.bin`).

## Directory Structure

```text
book/
├── source/         Source PGN / EPD opening suites (CCRL / human grandmaster lines)
├── builder/        Tools for compiling PGN games into polyglot binary books
└── README.md       Documentation and usage
```

## How to build an opening book

1. Place your PGN database in `book/source/openings.pgn`
2. Run the book builder tool:
   ```bash
   python3 builder/build_book.py --input source/openings.pgn --output ../book.bin --max_depth 20 --min_games 3
   ```
3. In UCI mode, set the book options:
   ```text
   setoption name OwnBook value true
   setoption name Book File value book.bin
   ```
