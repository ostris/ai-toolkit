import Loading from './Loading';
import classNames from 'classnames';

export interface TableColumn {
  title: string;
  key: string;
  render?: (row: any) => React.ReactNode;
  className?: string;
}

interface TableRow {
  [key: string]: any;
}

interface TableProps {
  columns: TableColumn[];
  rows: TableRow[];
  isLoading: boolean;
  onRefresh: () => void;
}

export default function UniversalTable({ columns, rows, isLoading, onRefresh = () => {} }: TableProps) {
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
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left text-gray-300">
            <thead className="text-xs uppercase bg-gray-800 text-gray-400">
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
                // Style for alternating rows
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
      )}
    </div>
  );
}
