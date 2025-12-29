import * as React from "react";
export type TileRowProps = React.PropsWithChildren<{
  name: string;
  style?: React.CSSProperties;
}>;
export default function TileRow(props: TileRowProps) {
  const rowId = React.useId();
  return (
    <div className="tile-row">
      <label htmlFor={rowId}>{props.name}</label>
      <div id={rowId} className="tile-row-content" style={props.style}>
      {props.children}
      </div>
    </div>
  );
}
