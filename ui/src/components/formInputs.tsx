import React from 'react';
import classNames from 'classnames';

const labelClasses = 'block text-xs mb-1 mt-2 text-gray-300';
const inputClasses =
  'w-full text-sm px-3 py-1 bg-gray-800 border border-gray-700 rounded-sm focus:ring-2 focus:ring-gray-600 focus:border-transparent';

export interface InputProps {
  label?: string;
  className?: string;
  placeholder?: string;
  required?: boolean;
}

export interface TextInputProps extends InputProps {
  value: string;
  onChange: (value: string) => void;
  type?: 'text' | 'password';
  disabled?: boolean;
}

export const TextInput = (props: TextInputProps) => {
  const { label, value, onChange, placeholder, required, disabled } = props;
  return (
    <div className={classNames(props.className)}>
      {label && <label className={labelClasses}>{label}</label>}
      <input
        type={props.type || 'text'}
        value={value}
        onChange={e => {
          if (disabled) return;
          onChange(e.target.value);
        }}
        className={`${inputClasses} ${disabled && 'opacity-30 cursor-not-allowed'}`}
        placeholder={placeholder}
        required={required}
        disabled={disabled}
      />
    </div>
  );
};

export interface NumberInputProps extends InputProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
}

export const NumberInput = (props: NumberInputProps) => {
  const { label, value, onChange, placeholder, required, min, max } = props;
  return (
    <div className={classNames(props.className)}>
      {label && <label className={labelClasses}>{label}</label>}
      <input
        type="number"
        value={value}
        onChange={e => {
          // Use parseFloat instead of Number to properly handle decimal values
          const rawValue = e.target.value;

          // Special handling for empty or partial inputs
          if (rawValue === '' || rawValue === '-' || rawValue === '.') {
            // For empty or partial inputs (like just a minus sign or decimal point),
            // we need to maintain the raw input in the input field
            // but pass a valid number to onChange
            onChange(0);
            return;
          }

          let value = Number(rawValue);

          // Handle NaN cases
          if (isNaN(value)) {
            value = 0;
          }

          // Apply min/max constraints only for valid numbers
          if (min !== undefined && value < min) value = min;
          if (max !== undefined && value > max) value = max;

          onChange(value);
        }}
        className={inputClasses}
        placeholder={placeholder}
        required={required}
        min={min}
        max={max}
        // Allow decimal points
        step="any"
      />
    </div>
  );
};

export interface SelectInputProps extends InputProps {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
}

export const SelectInput = (props: SelectInputProps) => {
  const { label, value, onChange, options } = props;
  return (
    <div className={classNames(props.className)}>
      {label && <label className={labelClasses}>{label}</label>}
      <select value={value} onChange={e => onChange(e.target.value)} className={inputClasses}>
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
};

export interface CheckboxProps {
  label?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  className?: string;
  required?: boolean;
  disabled?: boolean;
}

export const Checkbox = (props: CheckboxProps) => {
  const { label, checked, onChange, required, disabled } = props;
  const id = React.useId(); // Generate unique ID for label association

  return (
    <div className={classNames('flex items-center', props.className)}>
      <div className="relative flex items-start">
        <div className="flex items-center h-5">
          <input
            id={id}
            type="checkbox"
            checked={checked}
            onChange={e => onChange(e.target.checked)}
            className="w-4 h-4 rounded border-gray-700 bg-gray-800 text-indigo-600 focus:ring-2 focus:ring-indigo-500 focus:ring-offset-1 focus:ring-offset-gray-900 cursor-pointer transition-colors"
            required={required}
            disabled={disabled}
          />
        </div>
        {label && (
          <div className="ml-3 text-sm">
            <label
              htmlFor={id}
              className={classNames(
                'font-medium cursor-pointer select-none',
                disabled ? 'text-gray-500' : 'text-gray-300',
              )}
            >
              {label}
            </label>
          </div>
        )}
      </div>
    </div>
  );
};

interface FormGroupProps {
  label?: string;
  className?: string;
  children: React.ReactNode;
}

export const FormGroup: React.FC<FormGroupProps> = ({ label, className, children }) => {
  return (
    <div className={classNames(className)}>
      {label && <label className={labelClasses}>{label}</label>}
      <div className="px-4 space-y-2">{children}</div>
    </div>
  );
};
