import * as React from "react";

export type TileRowProps = React.PropsWithChildren<{
  name: string;
}>;
export default function TileRow(props: TileRowProps) {
  const rowId = React.useId();
  return (
    <div className="tile-row">
      <label htmlFor={rowId}>{props.name}</label>
      <div id={rowId} className="tile-row-content">
        {props.children}
      </div>
    </div>
  );
}
