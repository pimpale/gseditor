import { parse } from 'date-fns/parse';
import { enUS } from 'date-fns/locale/en-US';

function date_in_tz(date:Date, ianatz: string): Date {
  const date_in_tz_str = date.toLocaleString(enUS.code, { timeZone: ianatz });
  const date_in_tz = parse(date_in_tz_str , "MM/dd/yyyy, h:mm:ss aa", date, { locale: enUS });
  return date_in_tz;
}

export function getTimezoneOffset(ianatz: string): number {
  const now = new Date();
  const now_in_utc = date_in_tz(now, 'Europe/London');
  const now_in_tz = date_in_tz(now, ianatz);
  return now_in_tz.getTime() - now_in_utc.getTime();
}
