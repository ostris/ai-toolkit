'use client';

import { useState } from 'react';
import Loading from './Loading';
import classNames from 'classnames';
import { ChevronDown } from 'lucide-react';

export interface TableColumn {
  title: string;
  key: string;
  render?: (row: any) => React.ReactNode;
  mobileRender?: (row: any) => React.ReactNode;
  className?: string;
  hideOnMobile?: boolean;
  mobileAlignRight?: boolean;
}

interface TableRow {
  [key: string]: any;
}

interface TableProps {
  columns: TableColumn[];
  rows: TableRow[];
  isLoading: boolean;
  theadClassName?: string;
  onRefresh: () => void;
}

function MobileTable({ columns, rows }: { columns: TableColumn[]; rows: TableRow[] }) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const visibleCols = columns.filter(c => !c.hideOnMobile);
  const hiddenCols = columns.filter(c => c.hideOnMobile);

  return (
    <div className="divide-y divide-gray-700">
      {rows.map((row, index) => {
        const isExpanded = expandedIndex === index;
        const rowClass = index % 2 === 0 ? 'bg-gray-900' : 'bg-gray-800';
        return (
          <div key={index} className={rowClass}>
            <div
              className="flex items-center px-3 py-2 gap-2 cursor-pointer"
              onClick={(e) => {
                if ((e.target as HTMLElement).closest('a')) return;
                setExpandedIndex(isExpanded ? null : index);
              }}
            >
              {hiddenCols.length > 0 && (
                <ChevronDown className={classNames('w-4 h-4 text-gray-500 shrink-0 transition-transform', { 'rotate-180': isExpanded })} />
              )}
              {visibleCols.map(col => {
                const content = col.mobileRender ? col.mobileRender(row) : col.render ? col.render(row) : row[col.key];
                return (
                  <div key={col.key} className={classNames('text-sm', col.className, col.mobileAlignRight ? 'ml-auto' : 'min-w-0')}>
                    {content}
                  </div>
                );
              })}
            </div>
            {isExpanded && hiddenCols.length > 0 && (
              <div className="px-3 pb-3 pt-1 pl-9 space-y-2">
                {hiddenCols.map(col => (
                  <div key={col.key} className="flex items-center gap-2 text-sm">
                    <span className="text-xs uppercase text-gray-500 w-12 shrink-0">{col.title}</span>
                    <span className={col.className}>
                      {col.render ? col.render(row) : row[col.key]}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function UniversalTable({
  columns,
  rows,
  isLoading,
  theadClassName = 'text-gray-400',
  onRefresh = () => {},
}: TableProps) {
  return (
    <div className="w-full bg-gray-900 rounded-md shadow-md">
      {isLoading ? (
        <div className="p-4 flex justify-center">
          <Loading />
        </div>
      ) : rows.length === 0 ? (
        <div className="p-6 text-center text-gray-400">
          <p className="text-sm">Empty</p>
          <button
            onClick={() => onRefresh()}
            className="mt-2 px-3 py-1 text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 rounded transition-colors"
          >
            Refresh
          </button>
        </div>
      ) : (
        <>
          {/* Desktop table */}
          <div className="hidden md:block overflow-x-auto">
            <table className="w-full text-sm text-left text-gray-300">
              <thead className={classNames('text-xs uppercase bg-gray-800', theadClassName)}>
                <tr>
                  {columns.map(column => (
                    <th key={column.key} className="px-3 py-2">
                      {column.title}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows?.map((row, index) => {
                  const rowClass = index % 2 === 0 ? 'bg-gray-900' : 'bg-gray-800';
                  return (
                    <tr key={index} className={`${rowClass} border-b border-gray-700 hover:bg-gray-700`}>
                      {columns.map(column => (
                        <td key={column.key} className={classNames('px-3 py-2', column.className)}>
                          {column.render ? column.render(row) : row[column.key]}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {/* Mobile expandable rows */}
          <div className="md:hidden">
            <MobileTable columns={columns} rows={rows} />
          </div>
        </>
      )}
    </div>
  );
}
