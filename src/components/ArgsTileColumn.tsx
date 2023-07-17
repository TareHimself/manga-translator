import {
  EServerArgumentType,
  IServerArgument,
  IToServerArgument,
} from "../types";
import SelectTileRow from "./SelectTileRow";
import TileRow from "./TileRow";

export type ArgsTileColumnProps = {
  category: string;
  args: IToServerArgument[];
  argsInfo: IServerArgument[];
  onArgumentUpdated: (idx: number, update: string) => void;
};

export default function ArgsTileColumn(props: ArgsTileColumnProps) {
  return (
    <>
      {props.args.map((a, idx) => {
        const info = props.argsInfo[idx];
        const argumentName = props.category + " | " + a.name;
        const argumentKey = `${idx}`;
        if (info.type === EServerArgumentType.TEXT) {
          return (
            <TileRow name={argumentName} key={argumentKey}>
              <input
                type="text"
                onChange={(e) => props.onArgumentUpdated(idx, e.target.value)}
              />
            </TileRow>
          );
        } else if (info.type === EServerArgumentType.SELECT) {
          return (
            <SelectTileRow
              name={argumentName}
              key={argumentKey}
              value={info.options.findIndex((item) => item.value === a.value)}
              items={info.options.map((_, itemIdx) => itemIdx)}
              toSelectValue={(a) => `${a}`}
              toOriginalValue={parseInt}
              toLabel={(a) => info.options[a].name ?? "Loading"}
              onSelected={(a) => {
                props.onArgumentUpdated(idx, info.options[a].value);
              }}
            />
          );
        }
      })}
    </>
  );
}
