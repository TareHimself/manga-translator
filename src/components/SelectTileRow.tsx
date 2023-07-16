import TileRow from "./TileRow";

export type SelectTileRowProps<T> = {
  name: string;
  items: T[];
  value: T;
  toSelectValue: (value: T) => string;
  toOriginalValue: (value: string) => T;
  toLabel: (value: T) => string;
  onSelected: (value: T) => void;
  toKey?: (value: T) => React.Key;
};

export default function SelectTileRow<T>(props: SelectTileRowProps<T>) {
  return (
    <TileRow name={props.name}>
      <select
        name={props.name}
        onChange={(e) =>
          props.onSelected(props.toOriginalValue(e.target.value))
        }
        value={props.toSelectValue(props.value)}
      >
        {props.items.map((a) => (
          <option
            value={props.toSelectValue(a)}
            key={props.toKey ? props.toKey(a) : props.toSelectValue(a)}
          >
            {props.toLabel(a)}
          </option>
        ))}
      </select>
    </TileRow>
  );
}
