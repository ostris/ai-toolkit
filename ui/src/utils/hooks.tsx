import React from 'react';

/**
 * Updates a deeply nested value in an object using a string path
 * @param obj The object to update
 * @param value The new value to set
 * @param path String path to the property (e.g. 'config.process[0].model.name_or_path')
 * @returns A new object with the updated value
 */
export function setNestedValue<T, V>(obj: T, value: V, path?: string): T {
  // Create a copy of the original object to maintain immutability
  const result = { ...obj };

  // if path is not provided, be root path
  if (!path) {
    path = '';
  }

  // Split the path into segments
  const pathArray = path.split('.').flatMap(segment => {
    // Handle array notation like 'process[0]'
    const arrayMatch = segment.match(/^([^\[]+)(\[\d+\])+/);
    if (arrayMatch) {
      const propName = arrayMatch[1];
      const indices = segment
        .substring(propName.length)
        .match(/\[(\d+)\]/g)
        ?.map(idx => parseInt(idx.substring(1, idx.length - 1)));

      // Return property name followed by array indices
      return [propName, ...(indices || [])];
    }
    return segment;
  });

  // Navigate to the target location
  let current: any = result;
  for (let i = 0; i < pathArray.length - 1; i++) {
    const key = pathArray[i];

    // If current key is a number, treat it as an array index
    if (typeof key === 'number') {
      if (!Array.isArray(current)) {
        throw new Error(`Cannot access index ${key} of non-array`);
      }
      // Create a copy of the array to maintain immutability
      current = [...current];
    } else {
      // For object properties, create a new object if it doesn't exist
      if (current[key] === undefined) {
        // Check if the next key is a number, if so create an array, otherwise an object
        const nextKey = pathArray[i + 1];
        current[key] = typeof nextKey === 'number' ? [] : {};
      } else {
        // Create a shallow copy to maintain immutability
        current[key] = Array.isArray(current[key]) ? [...current[key]] : { ...current[key] };
      }
    }

    // Move to the next level
    current = current[key];
  }

  // Set the value at the final path segment
  const finalKey = pathArray[pathArray.length - 1];
  current[finalKey] = value;

  return result;
}

/**
 * Custom hook for managing a complex state object with string path updates
 * @param initialState The initial state object
 * @returns [state, setValue] tuple
 */
export function useNestedState<T>(initialState: T): [T, (value: any, path?: string) => void] {
  const [state, setState] = React.useState<T>(initialState);

  const setValue = React.useCallback((value: any, path?: string) => {
    if (path === undefined) {
      setState(value);
      return;
    }
    setState(prevState => setNestedValue(prevState, value, path));
  }, []);

  return [state, setValue];
}
