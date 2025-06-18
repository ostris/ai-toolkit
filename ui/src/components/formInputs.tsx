'use client';

import React, { forwardRef } from 'react';
import classNames from 'classnames';
import dynamic from 'next/dynamic';
import { CircleHelp } from 'lucide-react';
import { getDoc } from '@/docs';
import { openDoc } from '@/components/DocModal';

const Select = dynamic(() => import('react-select'), { ssr: false });

const labelClasses = 'block text-xs mb-1 mt-2 text-gray-300';
const inputClasses =
  'w-full text-sm px-3 py-1 bg-gray-800 border border-gray-700 rounded-sm focus:ring-2 focus:ring-gray-600 focus:border-transparent';

export interface InputProps {
  label?: string;
  docKey?: string;
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

export const TextInput = forwardRef<HTMLInputElement, TextInputProps>(
  ({ label, value, onChange, placeholder, required, disabled, type = 'text', className, docKey = null }, ref) => {
    const doc = getDoc(docKey);
    return (
      <div className={classNames(className)}>
        {label && (
          <label className={labelClasses}>
            {label}{' '}
            {doc && (
              <div className="inline-block ml-1 text-xs text-gray-500 cursor-pointer" onClick={() => openDoc(doc)}>
                <CircleHelp className="inline-block w-4 h-4 cursor-pointer" />
              </div>
            )}
          </label>
        )}
        <input
          ref={ref}
          type={type}
          value={value}
          onChange={e => {
            if (!disabled) onChange(e.target.value);
          }}
          className={`${inputClasses} ${disabled ? 'opacity-30 cursor-not-allowed' : ''}`}
          placeholder={placeholder}
          required={required}
          disabled={disabled}
        />
      </div>
    );
  },
);

// 👇 Helpful for debugging
TextInput.displayName = 'TextInput';

export interface NumberInputProps extends InputProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
}

export const NumberInput = (props: NumberInputProps) => {
  const { label, value, onChange, placeholder, required, min, max, docKey = null } = props;
  const doc = getDoc(docKey);

  // Add controlled internal state to properly handle partial inputs
  const [inputValue, setInputValue] = React.useState<string | number>(value ?? '');

  // Sync internal state with prop value
  React.useEffect(() => {
    setInputValue(value ?? '');
  }, [value]);

  return (
    <div className={classNames(props.className)}>
      {label && (
        <label className={labelClasses}>
          {label}{' '}
          {doc && (
            <div className="inline-block ml-1 text-xs text-gray-500 cursor-pointer" onClick={() => openDoc(doc)}>
              <CircleHelp className="inline-block w-4 h-4 cursor-pointer" />
            </div>
          )}
        </label>
      )}
      <input
        type="number"
        value={inputValue}
        onChange={e => {
          const rawValue = e.target.value;

          // Update the input display with the raw value
          setInputValue(rawValue);

          // Handle empty or partial inputs
          if (rawValue === '' || rawValue === '-') {
            // For empty or partial negative input, don't call onChange yet
            return;
          }

          const numValue = Number(rawValue);

          // Only apply constraints and call onChange when we have a valid number
          if (!isNaN(numValue)) {
            let constrainedValue = numValue;

            // Apply min/max constraints if they exist
            if (min !== undefined && constrainedValue < min) {
              constrainedValue = min;
            }
            if (max !== undefined && constrainedValue > max) {
              constrainedValue = max;
            }

            onChange(constrainedValue);
          }
        }}
        className={inputClasses}
        placeholder={placeholder}
        required={required}
        min={min}
        max={max}
        step="any"
      />
    </div>
  );
};

export interface SelectInputProps extends InputProps {
  value: string;
  disabled?: boolean;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
}

export const SelectInput = (props: SelectInputProps) => {
  const { label, value, onChange, options, docKey = null } = props;
  const doc = getDoc(docKey);
  const selectedOption = options.find(option => option.value === value);
  return (
    <div
      className={classNames(props.className, {
        'opacity-30 cursor-not-allowed': props.disabled,
      })}
    >
      {label && (
        <label className={labelClasses}>
          {label}{' '}
          {doc && (
            <div className="inline-block ml-1 text-xs text-gray-500 cursor-pointer" onClick={() => openDoc(doc)}>
              <CircleHelp className="inline-block w-4 h-4 cursor-pointer" />
            </div>
          )}
        </label>
      )}
      <Select
        value={selectedOption}
        options={options}
        isDisabled={props.disabled}
        className="aitk-react-select-container"
        classNamePrefix="aitk-react-select"
        onChange={selected => {
          if (selected) {
            onChange((selected as { value: string }).value);
          }
        }}
      />
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
  const id = React.useId();

  return (
    <div className={classNames('flex items-center gap-3', props.className)}>
      <button
        type="button"
        role="switch"
        id={id}
        aria-checked={checked}
        aria-required={required}
        disabled={disabled}
        onClick={() => !disabled && onChange(!checked)}
        className={classNames(
          'relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-600 focus:ring-offset-2',
          checked ? 'bg-blue-600' : 'bg-gray-700',
          disabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-opacity-80',
        )}
      >
        <span className="sr-only">Toggle {label}</span>
        <span
          className={classNames(
            'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out',
            checked ? 'translate-x-5' : 'translate-x-0',
          )}
        />
      </button>
      {label && (
        <label
          htmlFor={id}
          className={classNames(
            'text-sm font-medium cursor-pointer select-none',
            disabled ? 'text-gray-500' : 'text-gray-300',
          )}
        >
          {label}
        </label>
      )}
    </div>
  );
};

interface FormGroupProps {
  label?: string;
  className?: string;
  docKey?: string;
  children: React.ReactNode;
}

export const FormGroup: React.FC<FormGroupProps> = ({ label, className, children, docKey = null }) => {
  const doc = getDoc(docKey);
  return (
    <div className={classNames(className)}>
      {label && (
        <label className={labelClasses}>
          {label}{' '}
          {doc && (
            <div className="inline-block ml-1 text-xs text-gray-500 cursor-pointer" onClick={() => openDoc(doc)}>
              <CircleHelp className="inline-block w-4 h-4 cursor-pointer" />
            </div>
          )}
        </label>
      )}
      <div className="px-4 space-y-2">{children}</div>
    </div>
  );
};
