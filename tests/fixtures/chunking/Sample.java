package demo;

import java.util.List;

/** Inventory of items. */
public class Inventory {
    private final List<String> items;

    public Inventory(List<String> items) {
        this.items = items;
    }

    // Counts items.
    public int count() {
        return items.size();
    }

    enum Kind { SMALL, LARGE }
}
