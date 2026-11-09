import React from "react";

type Props = { title: string };

export function Header({ title }: Props) {
  return <h1>{title}</h1>;
}

export const Footer = () => <footer>bye</footer>;
