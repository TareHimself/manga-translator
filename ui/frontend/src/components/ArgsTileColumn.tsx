import { NumberInput, Switch, TextInput , type ComboboxData, type ComboboxItem,  Select } from "@mantine/core";
import {
  EServerArgumentType,
  type IPluginArgument,
  type IPluginArgumentInfo,
  type IPluginSelectArgument,
} from "../types";
import { useMemo } from "react";
export type SelectArgumentProps = {
  label: string
  description: string
  defaultValue: string
  options: IPluginSelectArgument[]
  value?: string
  onChange: (selected: string) => void
}

function SelectArgument(props: SelectArgumentProps) {
  const data = useMemo<ComboboxData>(
    () =>
      props.options.map<ComboboxItem>((c) => ({
        label: c.name,
        value: c.value,
      })),
    [props]
  );

  return (
    <Select
      allowDeselect={false}
      maxLength={30}
      searchable={true}
      label={props.label}
      description={props.description}
      data={data}
      value={props.value ?? props.defaultValue}
      onChange={(value) => props.onChange(value ?? props.options[0].value)}
      style={{ width: '60%' }}
    />
  );
}

export type ArgsTileColumnProps = {
  category: string;
  args: IPluginArgumentInfo[];
  argsInfo: IPluginArgument[];
  onArgumentUpdated: (idx: number, update: unknown) => void;
};

export default function ArgsTileColumn(props: ArgsTileColumnProps) {
  return (
    <>
      {props.args.map((_, idx) => {
        const info = props.argsInfo[idx];
        const argumentName = props.category + " | " + info.name;
        const argumentKey = `${info.id}`;
        if (info.type === EServerArgumentType.STRING) {
          return (
            <TextInput
              key={argumentKey}
              label={argumentName}
              description={info.description}
              defaultValue={info.default}
              value={props.args[idx].value as string ?? info.default}
              onChange={(e) => props.onArgumentUpdated(idx, e.currentTarget.value)}
              style={{ width: "60%" }}
            />
          );
        } else if (info.type === EServerArgumentType.INT) {
          return (
            <NumberInput
              key={argumentKey}
              label={argumentName}
              description={info.description}
              defaultValue={info.default}
              value={props.args[idx].value as number ?? info.default}
              onChange={(e) => props.onArgumentUpdated(idx,typeof e === "string" ? parseInt(e) : e)}
              style={{ width: "60%" }}
              allowDecimal={false}
            />
          );
        } else if (info.type === EServerArgumentType.BOOLEAN) {
          return (
            <Switch
              key={argumentKey}
              label={argumentName}
              description={info.description}
              defaultChecked={info.default}
              checked={props.args[idx].value as boolean ?? info.default}
              onChange={(e) => props.onArgumentUpdated(idx, e.currentTarget.checked)}
              style={{ width: "60%" }}
              labelPosition="left"
            />
          );
        } else if (info.type === EServerArgumentType.SELECT) {
          return (
            <SelectArgument
              label={argumentName}
              key={argumentKey}
              description={info.description}
              defaultValue={info.default}
              value={props.args[idx].value as string ?? info.default}
              options={info.options}
              onChange={(a) => {
                props.onArgumentUpdated(idx,a);
              }}
            />
          );
        }

        return <></>;
      })}
    </>
  );
}
