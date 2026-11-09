import fs from "fs";

// Reads a config file.
export function readConfig(path) {
  return JSON.parse(fs.readFileSync(path, "utf8"));
}

export const double = (x) => x * 2;

class Counter {
  constructor() {
    this.n = 0;
  }

  increment() {
    this.n += 1;
  }
}

module.exports = { Counter };
