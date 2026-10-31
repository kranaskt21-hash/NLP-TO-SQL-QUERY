import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from nl2sql.db import schema
FUNCTIONS = set('abs avg count sum total min max round coalesce ifnull nullif lower upper length substr substring trim ltrim rtrim replace instr like glob date time datetime julianday strftime unixepoch current_date current_time current_timestamp cast row_number rank dense_rank lag lead first_value last_value nth_value ntile percent_rank cume_dist group_concat printf iif typeof unicode char hex random'.split())
class ValidationError(ValueError): pass
def validated_tree(sql):
    try:
        trees = sqlglot.parse(sql,read='sqlite')
        if len(trees)!=1 or trees[0] is None: raise ValidationError('Exactly one query is required')
        tree=trees[0]
        if not isinstance(tree,(exp.Select,exp.Union,exp.Intersect,exp.Except)): raise ValidationError('Only SELECT queries allowed')
        forbidden = (exp.Insert,exp.Update,exp.Delete,exp.Create,exp.Drop,exp.Command,exp.Pragma,exp.Attach,exp.Detach,exp.Into,exp.Transaction)
        if any(isinstance(n,forbidden) for n in tree.walk()): raise ValidationError('Prohibited statement')
        for f in tree.find_all(exp.Func):
            name = f.name if isinstance(f,exp.Anonymous) else f.sql_name()
            # SQLGlot canonicalizes SQLite spellings.
            if name.lower() not in FUNCTIONS | {'time_to_str','ts_or_ds_to_timestamp','str_to_time','ts_or_ds_to_date','timestamp_trunc','concat','extract','and','or','case','if','exists'}:
                raise ValidationError('Prohibited function: '+name)
        return qualify(tree,dialect='sqlite',schema=schema(),validate_qualify_columns=True,identify=False)
    except ValidationError: raise
    except Exception as exc: raise ValidationError('Invalid SQL or unavailable schema reference: '+str(exc)[:250]) from None
def validate(sql):
    validated_tree(sql)
    return sql

def references(sql):
    """Resolve base tables/columns via qualified scopes; derived aliases are not labels."""
    from sqlglot.optimizer.scope import traverse_scope, Scope
    try:
        tree=validated_tree(sql); tables=set(); columns=set()
        for scope in traverse_scope(tree):
            sources=scope.sources
            for source in sources.values():
                if isinstance(source,exp.Table): tables.add(source.name)
            for col in scope.columns:
                source=sources.get(col.table)
                if isinstance(source,exp.Table): columns.add(source.name+'.'+col.name)
        return {'tables':sorted(tables),'columns':sorted(columns),'supported':True}
    except Exception as exc:
        return {'tables':[],'columns':[],'supported':False,'reason':str(exc)}
