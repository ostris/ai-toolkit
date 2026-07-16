// Minimal terminal emulator for rendering job logs the way a real terminal
// would. Handles the control characters and ANSI escape sequences that show up
// in training logs on Linux, macOS, and Windows:
//   \r      carriage return  — cursor to column 0, output overwrites in place
//   \n      newline          — next line (also covers \r\n)
//   \b      backspace        — cursor left one column
//   \t      tab              — advance to next 8-column tab stop
//   ESC[A / ESC[B / ESC[C / ESC[D ... cursor movement (tqdm nested bars)
//   ESC[K   erase in line, ESC[J erase in display
//   ESC[m   colors/styles    — stripped (the UI renders plain text)
//   OSC sequences (window title etc.) — stripped
// It is stateful and incremental: feed it chunks as they arrive and it keeps
// cursor position and partial escape sequences across chunk boundaries.

// Cap the scrollback so multi-day jobs don't grow memory without bound.
const MAX_LINES = 5000;
// If an escape sequence is never terminated, stop buffering it after this.
const MAX_PENDING = 4096;

export class TerminalEmulator {
  private lines: string[] = [''];
  private row = 0;
  private col = 0;
  // Incomplete escape sequence carried over to the next write().
  private pending = '';

  reset(): void {
    this.lines = [''];
    this.row = 0;
    this.col = 0;
    this.pending = '';
  }

  write(chunk: string): void {
    const text = this.pending + chunk;
    this.pending = '';
    const n = text.length;
    let i = 0;
    while (i < n) {
      const code = text.charCodeAt(i);
      if (code === 0x1b) {
        const consumed = this.consumeEscape(text, i);
        if (consumed === 0) {
          // Sequence is split across chunks — save it for the next write.
          if (n - i <= MAX_PENDING) {
            this.pending = text.slice(i);
          }
          return;
        }
        i += consumed;
      } else if (code === 0x0d) {
        // \r
        this.col = 0;
        i++;
      } else if (code === 0x0a) {
        // \n
        this.newline();
        i++;
      } else if (code === 0x08) {
        // \b
        if (this.col > 0) this.col--;
        i++;
      } else if (code === 0x09) {
        // \t
        this.putText(' '.repeat(8 - (this.col % 8)));
        i++;
      } else if (code < 0x20 || code === 0x7f) {
        // Other control chars (BEL, etc.) — ignore.
        i++;
      } else {
        // Bulk-copy a run of printable characters.
        let j = i + 1;
        while (j < n) {
          const c = text.charCodeAt(j);
          if (c < 0x20 || c === 0x7f) break;
          j++;
        }
        this.putText(text.slice(i, j));
        i = j;
      }
    }
  }

  /** The rendered screen/scrollback as plain text lines. */
  toLines(): string[] {
    return this.lines.slice();
  }

  toString(): string {
    return this.lines.join('\n');
  }

  // Overwrite text at the cursor, padding with spaces if the cursor sits past
  // the end of the line (just like a real terminal cell grid).
  private putText(s: string): void {
    let line = this.lines[this.row];
    if (line.length < this.col) {
      line = line.padEnd(this.col);
    }
    this.lines[this.row] = line.slice(0, this.col) + s + line.slice(this.col + s.length);
    this.col += s.length;
  }

  private newline(): void {
    this.row++;
    this.col = 0;
    this.ensureRow();
    const excess = this.lines.length - MAX_LINES;
    if (excess > 0) {
      this.lines.splice(0, excess);
      this.row = Math.max(0, this.row - excess);
    }
  }

  private ensureRow(): void {
    while (this.lines.length <= this.row) {
      this.lines.push('');
    }
  }

  // Consume one escape sequence starting at text[i] (which is ESC). Returns
  // the number of characters consumed, or 0 if the sequence is incomplete.
  private consumeEscape(text: string, i: number): number {
    const n = text.length;
    if (i + 1 >= n) return 0;
    const kind = text[i + 1];

    if (kind === '[') {
      // CSI: ESC [ <params> <intermediates> <final byte @-~>
      let j = i + 2;
      while (j < n && /[0-9;?]/.test(text[j])) j++;
      while (j < n && text[j] >= ' ' && text[j] <= '/') j++;
      if (j >= n) return 0;
      const final = text[j];
      if (final >= '@' && final <= '~') {
        this.applyCsi(text.slice(i + 2, j), final);
        return j - i + 1;
      }
      // Malformed — drop the ESC [ and let the rest render as text.
      return 2;
    }

    if (kind === ']') {
      // OSC: ESC ] ... terminated by BEL or ST (ESC \)
      for (let j = i + 2; j < n; j++) {
        if (text.charCodeAt(j) === 0x07) return j - i + 1;
        if (text.charCodeAt(j) === 0x1b) {
          if (j + 1 >= n) return 0;
          if (text[j + 1] === '\\') return j - i + 2;
        }
      }
      return 0;
    }

    if (kind === 'M') {
      // Reverse index — cursor up one line.
      this.row = Math.max(0, this.row - 1);
      return 2;
    }

    // Any other two-character escape (ESC 7, ESC 8, ESC c, ...) — ignore.
    return 2;
  }

  private applyCsi(params: string, final: string): void {
    const args = params.replace(/^\?/, '').split(';').map(p => parseInt(p, 10));
    const arg = (idx: number, fallback: number): number => {
      const v = args[idx];
      return Number.isNaN(v) || v === undefined ? fallback : v;
    };

    switch (final) {
      case 'A': // cursor up
      case 'F': // cursor to start of previous line
        this.row = Math.max(0, this.row - arg(0, 1));
        if (final === 'F') this.col = 0;
        break;
      case 'B': // cursor down
      case 'e':
      case 'E': // cursor to start of next line
        this.row += arg(0, 1);
        this.ensureRow();
        if (final === 'E') this.col = 0;
        break;
      case 'C': // cursor forward
      case 'a':
        this.col += arg(0, 1);
        break;
      case 'D': // cursor back
        this.col = Math.max(0, this.col - arg(0, 1));
        break;
      case 'G': // cursor to absolute column (1-based)
      case '`':
        this.col = Math.max(0, arg(0, 1) - 1);
        break;
      case 'K': {
        // Erase in line: 0 = cursor to end, 1 = start to cursor, 2 = all
        const mode = arg(0, 0);
        const line = this.lines[this.row];
        if (mode === 0) {
          this.lines[this.row] = line.slice(0, this.col);
        } else if (mode === 1) {
          this.lines[this.row] = ' '.repeat(Math.min(this.col + 1, line.length)) + line.slice(this.col + 1);
        } else if (mode === 2) {
          this.lines[this.row] = '';
        }
        break;
      }
      case 'J': {
        // Erase in display: 0 = cursor to end, 2/3 = everything
        const mode = arg(0, 0);
        if (mode === 0) {
          this.lines[this.row] = this.lines[this.row].slice(0, this.col);
          this.lines.length = this.row + 1;
        } else if (mode === 2 || mode === 3) {
          this.reset();
        }
        break;
      }
      // Colors ('m'), cursor show/hide ('h'/'l'), positioning we can't map to
      // scrollback ('H'/'f'/'d'), and anything else — ignore.
      default:
        break;
    }
  }
}
