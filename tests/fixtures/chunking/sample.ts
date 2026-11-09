import { readFile } from "fs/promises";

export interface User {
  id: number;
  name: string;
}

export type Id = string | number;

/** Loads a user. */
export async function loadUser(id: Id): Promise<User> {
  const raw = await readFile(`${id}.json`, "utf8");
  return JSON.parse(raw);
}

export class UserService {
  private cache = new Map<Id, User>();

  get(id: Id): User | undefined {
    return this.cache.get(id);
  }
}

const toName = (u: User): string => u.name;
