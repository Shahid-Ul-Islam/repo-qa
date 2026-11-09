package store

import "fmt"

// Item is a stored item.
type Item struct {
	ID   int
	Name string
}

// NewItem builds an item.
func NewItem(id int, name string) *Item {
	return &Item{ID: id, Name: name}
}

// Describe formats the item.
func (i *Item) Describe() string {
	return fmt.Sprintf("%d:%s", i.ID, i.Name)
}
