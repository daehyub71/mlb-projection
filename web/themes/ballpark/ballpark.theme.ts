// The ballpark theme. `astryx experimental theme editor` wrote this file once and never changes it again.
// The checked settings live in ballpark.settings.json and the generated ballpark.settings.ts;
// change them with `astryx experimental theme editor set ballpark ...`. Add your own changes below: they win over the settings.
import {defineTheme} from '@astryxdesign/core/theme';
import {ballparkSettings} from './ballpark.settings';

export const ballparkTheme = defineTheme({
  extends: ballparkSettings,
  name: 'ballpark',
});
