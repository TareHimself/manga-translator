import { IToServerArgument } from "../types";
import TileRow from "./TileRow";

export type ArgsTileColumnProps = {
  category: string;
  args: IToServerArgument[];
  onArgumentUpdated: (idx: number, update: string) => void;
};

export default function ArgsTileColumn(props: ArgsTileColumnProps) {
  return (
    <>
      {props.args.map((a, idx) => (
        <TileRow name={props.category + " | " + a.name} key={`${idx}`}>
          <input
            type="text"
            onChange={(e) => props.onArgumentUpdated(idx, e.target.value)}
          />
        </TileRow>
      ))}
    </>
  );
}
