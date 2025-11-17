'use client';

import React, { forwardRef } from 'react';
import classNames from 'classnames';
import dynamic from 'next/dynamic';
import { CircleHelp } from 'lucide-react';
import { getDoc } from '@/docs';
import { openDoc } from '@/components/DocModal';
import { ConfigDoc, GroupedSelectOption, SelectOption } from '@/types';

const Select = dynamic(() => import('react-select'), { ssr: false });

const labelClasses = 'block text-xs mb-1 mt-2 text-gray-300';
const inputClasses =
  'w-full text-sm px-3 py-1 bg-gray-800 border border-gray-700 rounded-sm focus:ring-2 focus:ring-gray-600 focus:border-transparent';

export interface InputProps {
  label?: string;
  docKey?: string | null;
  doc?: ConfigDoc | null;
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

export const TextInput = forwardRef<HTMLInputElement, TextInputProps>((props: TextInputProps, ref) => {
  const { label, value, onChange, placeholder, required, disabled, type = 'text', className, docKey = null } = props;
  let { doc } = props;
  if (!doc && docKey) {
    doc = getDoc(docKey);
  }

  // Add controlled internal state to handle rapid typing and avoid stale closure issues
  const [inputValue, setInputValue] = React.useState<string>(value ?? '');

  // Sync internal state with prop value when it changes externally
  React.useEffect(() => {
    setInputValue(value ?? '');
  }, [value]);

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
        value={inputValue}
        onChange={e => {
          if (!disabled) {
            const newValue = e.target.value;
            setInputValue(newValue);
            onChange(newValue);
          }
        }}
        className={`${inputClasses} ${disabled ? 'opacity-30 cursor-not-allowed' : ''}`}
        placeholder={placeholder}
        required={required}
        disabled={disabled}
      />
    </div>
  );
});

// 👇 Helpful for debugging
TextInput.displayName = 'TextInput';

export interface NumberInputProps extends InputProps {
  value: number | null;
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  step?: number | string;
}

export const NumberInput = (props: NumberInputProps) => {
  const { label, value, onChange, placeholder, required, min, max, step = 'any', docKey = null } = props;
  let { doc } = props;
  if (!doc && docKey) {
    doc = getDoc(docKey);
  }

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
        step={step}
      />
    </div>
  );
};

export interface SelectInputProps extends InputProps {
  value: string;
  disabled?: boolean;
  onChange: (value: string) => void;
  options: GroupedSelectOption[] | SelectOption[];
}

export const SelectInput = (props: SelectInputProps) => {
  const { label, value, onChange, options, docKey = null } = props;
  let { doc } = props;
  if (!doc && docKey) {
    doc = getDoc(docKey);
  }
  let selectedOption: SelectOption | undefined;
  if (options && options.length > 0) {
    // see if grouped options
    if ('options' in options[0]) {
      selectedOption = (options as GroupedSelectOption[])
        .flatMap(group => group.options)
        .find(opt => opt.value === value);
    } else {
      selectedOption = (options as SelectOption[]).find(opt => opt.value === value);
    }
  }
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
  label?: string | React.ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
  className?: string;
  required?: boolean;
  disabled?: boolean;
  docKey?: string | null;
  doc?: ConfigDoc | null;
}

export const Checkbox = (props: CheckboxProps) => {
  const { label, checked, onChange, required, disabled } = props;
  let { doc } = props;
  if (!doc && props.docKey) {
    doc = getDoc(props.docKey);
  }

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
        <>
          <label
            htmlFor={id}
            className={classNames(
              'text-sm font-medium cursor-pointer select-none',
              disabled ? 'text-gray-500' : 'text-gray-300',
            )}
          >
            {label}
          </label>
          {doc && (
            <div className="inline-block ml-1 text-xs text-gray-500 cursor-pointer" onClick={() => openDoc(doc)}>
              <CircleHelp className="inline-block w-4 h-4 cursor-pointer" />
            </div>
          )}
        </>
      )}
    </div>
  );
};

interface FormGroupProps {
  label?: string;
  className?: string;
  docKey?: string | null;
  doc?: ConfigDoc | null;
  tooltip?: string;
  children: React.ReactNode;
}

export const FormGroup: React.FC<FormGroupProps> = props => {
  const { label, className, children, docKey = null, tooltip } = props;
  let { doc } = props;
  if (!doc && docKey) {
    doc = getDoc(docKey);
  }
  // If tooltip is provided but no doc, create a simple doc object
  const effectiveDoc = doc || (tooltip ? { title: label || '', description: tooltip } : null);
  return (
    <div className={classNames(className)}>
      {label && (
        <label className={classNames(labelClasses, 'mb-2')}>
          {label}{' '}
          {effectiveDoc && (
            <div className="inline-block ml-1 text-xs text-gray-500 cursor-pointer" onClick={() => openDoc(effectiveDoc)}>
              <CircleHelp className="inline-block w-4 h-4 cursor-pointer" />
            </div>
          )}
        </label>
      )}
      <div className="space-y-2">{children}</div>
    </div>
  );
};

export interface SliderInputProps extends InputProps {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  showValue?: boolean;
}

export const SliderInput: React.FC<SliderInputProps> = props => {
  const { label, value, onChange, min, max, step = 1, disabled, className, docKey = null, showValue = true } = props;
  let { doc } = props;
  if (!doc && docKey) {
    doc = getDoc(docKey);
  }

  const trackRef = React.useRef<HTMLDivElement | null>(null);
  const [dragging, setDragging] = React.useState(false);

  const clamp = (v: number) => (v < min ? min : v > max ? max : v);
  const snapToStep = (v: number) => {
    if (!Number.isFinite(v)) return min;
    const steps = Math.round((v - min) / step);
    const snapped = min + steps * step;
    return clamp(Number(snapped.toFixed(6)));
  };

  const percent = React.useMemo(() => {
    if (max === min) return 0;
    const p = ((value - min) / (max - min)) * 100;
    return p < 0 ? 0 : p > 100 ? 100 : p;
  }, [value, min, max]);

  const calcFromClientX = React.useCallback(
    (clientX: number) => {
      const el = trackRef.current;
      if (!el || !Number.isFinite(clientX)) return;
      const rect = el.getBoundingClientRect();
      const width = rect.right - rect.left;
      if (!(width > 0)) return;

      // Clamp ratio to [0, 1] so it can never flip ends.
      const ratioRaw = (clientX - rect.left) / width;
      const ratio = ratioRaw <= 0 ? 0 : ratioRaw >= 1 ? 1 : ratioRaw;

      const raw = min + ratio * (max - min);
      onChange(snapToStep(raw));
    },
    [min, max, step, onChange],
  );

  // Mouse/touch pointer drag
  const onPointerDown = (e: React.PointerEvent) => {
    if (disabled) return;
    e.preventDefault();

    // Capture the pointer so moves outside the element are still tracked correctly
    try {
      (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
    } catch {}

    setDragging(true);
    calcFromClientX(e.clientX);

    const handleMove = (ev: PointerEvent) => {
      ev.preventDefault();
      calcFromClientX(ev.clientX);
    };
    const handleUp = (ev: PointerEvent) => {
      setDragging(false);
      // release capture if we got it
      try {
        (e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId);
      } catch {}
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
    };

    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
  };

  return (
    <div className={classNames(className, disabled ? 'opacity-30 cursor-not-allowed' : '')}>
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

      <div className="flex items-center gap-3">
        <div className="flex-1">
          <div
            ref={trackRef}
            onPointerDown={onPointerDown}
            className={classNames(
              'relative w-full h-6 select-none outline-none',
              disabled ? 'pointer-events-none' : 'cursor-pointer',
            )}
          >
            {/* Thicker track */}
            <div className="pointer-events-none absolute left-0 right-0 top-1/2 -translate-y-1/2 h-3 rounded-sm bg-gray-800 border border-gray-700" />

            {/* Fill */}
            <div
              className="pointer-events-none absolute left-0 top-1/2 -translate-y-1/2 h-3 rounded-sm bg-blue-600"
              style={{ width: `${percent}%` }}
            />

            {/* Thumb */}
            <div
              onPointerDown={onPointerDown}
              className={classNames(
                'absolute top-1/2 -translate-y-1/2 -ml-2',
                'h-4 w-4 rounded-full bg-white shadow border border-gray-300 cursor-pointer',
                'after:content-[""] after:absolute after:inset-[-6px] after:rounded-full after:bg-transparent', // expands hit area
                dragging ? 'ring-2 ring-blue-600' : '',
              )}
              style={{ left: `calc(${percent}% )` }}
            />
          </div>

          <div className="flex justify-between text-xs text-gray-500 mt-0.5 select-none">
            <span>{min}</span>
            <span>{max}</span>
          </div>
        </div>

        {showValue && (
          <div className="min-w-[3.5rem] text-right text-sm px-3 py-1 bg-gray-800 border border-gray-700 rounded-sm">
            {Number.isFinite(value) ? value : ''}
          </div>
        )}
      </div>
    </div>
  );
};
