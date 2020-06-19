'''
This script online document public APIs from modules which is listed in `__all__` attribute
Reference : https://pdoc3.github.io/pdoc/doc/pdoc/#gsc.tab=0
'''

import inspect
from docstring_parser import parse
import docstring_parser

PUBLIC_API_PACKAGE_LIST = [
    'vortex.core.pipelines',
    # 'vortex.core.factory'
    ]

DOCS_TEMPLATE = '''# {}

---

---

{}

{}

'''

CLASS_TEMPLATE = '''## Classes

---

'''

FUNCTION_TEMPLATE = '''## Functions

---

'''


SUBHEADING_TEMPLATE = '''---

### {}

'''

SUBHEADING_DESC_TEMPLATE = '''

{}

'''

CLASS_FUNCTION_START_TEMPLATE = '''#### `{}`

'''

FUNCTION_START_TEMPLATE = '''

```python
def {}(
'''

FUNCTION_ARGS_TEMPLATE = '''      {}{}{},
'''

FUNCTION_END_TEMPLATE = ''')
```

'''

FUNCTION_PART_START_TEMPLATE = '''

**{}**:

'''

FUNCTION_ARG_TEMPLATE = '''- `{}`{}- {}
'''

FUNCTION_RETURN_TEMPLATE = '''- {} - {}
'''

FUNCTION_RAISES_TEMPLATE = '''- {} - {}
'''

FUNCTION_META_TEMPLATE='''

{}

'''


DIVISION_SEPARATION = '''

---

'''

def extract_function_docs(md_template,function):
    md_template+=FUNCTION_START_TEMPLATE.format(function.__name__)
    # function signature
    arg_parameters = inspect.signature(function).parameters
    for arg in arg_parameters.keys():
        parameter = arg_parameters[arg]
        arg_name = parameter.name
        if parameter.kind == parameter.VAR_KEYWORD:
            arg_name = '**'+arg_name
        if parameter.kind == parameter.VAR_POSITIONAL:
            arg_name = '*'+arg_name
        arg_annotations = parameter.annotation
        if arg_annotations == inspect._empty:
            arg_annotations = ''
        else:
            arg_annotations = str(arg_annotations)
            if len(arg_annotations.split('\'')) > 1:
                arg_annotations = arg_annotations.split('\'')[1]
            arg_annotations = ' : {}'.format(arg_annotations)
        default = parameter.default
        if default == inspect._empty:
            default = ''
        else:
            if isinstance(default,str):
                default="'{}'".format(default)
            default = ' = {}'.format(default)
        md_template+=FUNCTION_ARGS_TEMPLATE.format(arg_name,arg_annotations,default)
        
    md_template+=FUNCTION_END_TEMPLATE
    
    # Extract docstring
    docs = inspect.getdoc(function)
    docs = parse(docs)

    ## Extract arguments docstring
    arg_template=''
    for arg in docs.params :
        if arg_template=='':
            md_template+=FUNCTION_PART_START_TEMPLATE.format('Arguments')
            arg_template=FUNCTION_PART_START_TEMPLATE.format('Arguments')
        argname=arg.arg_name
        type_name=''
        if arg.type_name is not None and arg.type_name != '[type]':
            type_name=' _{}'.format(arg.type_name)
        if arg.is_optional:
            if type_name=='':
                type_name=' _optional_ '
            else:
                type_name+=', optional_ '
        elif type_name!= '':
            type_name+='_ '
        argdesc=arg.description
        md_template+=FUNCTION_ARG_TEMPLATE.format(argname,type_name,argdesc)

    ## Extract returns docstring
    
    returns = docs.returns
    if returns is not None:
        md_template+=FUNCTION_PART_START_TEMPLATE.format('Returns')
        return_type = returns.type_name
        return_desc = returns.description
        
        md_template+=FUNCTION_RETURN_TEMPLATE.format('`{}`'.format(return_type),return_desc)

    ## Extract raises docstring
    raises = docs.raises
    raise_template = ''
    for err_raise in raises:
        if raise_template=='':
            md_template+=FUNCTION_PART_START_TEMPLATE.format('Raises')
            raise_template=FUNCTION_PART_START_TEMPLATE.format('Raises')
        raise_type = err_raise.type_name
        raise_desc = err_raise.description
        
        md_template+=FUNCTION_RAISES_TEMPLATE.format('`{}`'.format(raise_type),raise_desc)

    ## Extract example
    for meta in docs.meta:
        if meta.args[0]=='example' or meta.args[0]=='examples':
            md_template+=FUNCTION_PART_START_TEMPLATE.format( meta.args[0].title())
            md_template+=FUNCTION_META_TEMPLATE.format(meta.description)

    md_template+=DIVISION_SEPARATION
    return md_template

def class_docs_handler(md_template,api):
    if len(md_template)==0:
        md_template+=CLASS_TEMPLATE
    md_template += SUBHEADING_TEMPLATE.format(api.__name__)

    desc = inspect.getdoc(api)
    md_template+=SUBHEADING_DESC_TEMPLATE.format(desc)
    member_function = inspect.getmembers(api,inspect.isfunction)
    
    # Iterate all class function without '_' suffix (except __init__)
    for function in member_function:
        if function[0].startswith('_') and function[0]!='__init__':
            continue
        func_name = function[0]
        md_template+=CLASS_FUNCTION_START_TEMPLATE.format(func_name)
        
        md_template=extract_function_docs(md_template,function[1])


    return md_template

def function_docs_handler(md_template,api):
    if len(md_template)==0:
        md_template+=FUNCTION_TEMPLATE
    md_template += SUBHEADING_TEMPLATE.format(api.__name__)
    docs = inspect.getdoc(api)
    docs = parse(docs)
    short_desc = docs.short_description

    md_template+=SUBHEADING_DESC_TEMPLATE.format(short_desc)
    md_template=extract_function_docs(md_template,api)

    return md_template
    
def main():
    for package in PUBLIC_API_PACKAGE_LIST:
        # md_template = HEADING_TEMPLATE.format(package)
        class_md_template = ''
        function_md_template = ''

        exec('import {}'.format(package))
        
        if eval(package).__file__.split('/')[-1] != '__init__.py':
            public_apis = eval(package).__all__
            for api in public_apis:
                exec('from {} import {}'.format(package,api))
                api = eval(api)

                # Handler if api is a class
                if inspect.isclass(api):
                    class_md_template=class_docs_handler(class_md_template,api)

                # Handler if api is a function
                elif inspect.isfunction(api):
                    function_md_template=function_docs_handler(function_md_template,api)
        else:
            # Only extract submodule under specified package directory
            submodules = inspect.getmembers(eval(package), inspect.ismodule)
            submodules = [module for module in submodules if module[1].__name__.startswith(package)]

            # Iterate all submodules
            for module in sorted(submodules):
                if hasattr(module[1],'__all__'):
                    public_apis = module[1].__all__

                    # Iterate all API exist in __all__ on all submodule
                    for api in public_apis:
                        exec('from {}.{} import {}'.format(package,module[0],api))
                        api_name = api
                        api = eval(api)

                        # Handler if api is a class
                        if inspect.isclass(api):
                            class_md_template=class_docs_handler(class_md_template,api)

                        # Handler if api is a function
                        elif inspect.isfunction(api):
                            function_md_template=function_docs_handler(function_md_template,api)

        md_template = DOCS_TEMPLATE.format(package,class_md_template,function_md_template)

        with open('docs/api/{}.md'.format(package),'w') as f:
            f.write(md_template)

if __name__ == "__main__":
    main()
